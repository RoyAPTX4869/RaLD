"""Radar Digital Signal Processing.

This radar signal processing module provides the necessary tools to
process the raw IQ measurements of a MIMO radar sensor into exploitable
pointcloud and heatmap.

NOTE: Make sure that a calibration stage is applied to the raw ADC data
before further processing.
"""
import numpy as np


# Speed of light
C: float = 299792458.0
# Minimum number of szimuth bins
NUMBER_AZIMUTH_BINS_MIN: int = 32

# Minimum number of elevation bins
NUMBER_ELEVATION_BINS_MIN: int = 32

# Minimum number of doppler bins
NUMBER_DOPPLER_BINS_MIN: int = 16

# Minimum number of range bins
NUMBER_RANGE_BINS_MIN: int = 128



def steering_matrix(txl: np.array, rxl: np.array,
                            az: np.array, el: np.array,) -> np.array:
    """Build a steering matrix.

    Arguments:
        txl: TX Antenna layout
        rxl: RX Antenna layout
        az: Azimuth angle
        el: Elevation angle
    """
    taz = txl[:, 1]
    tel = txl[:, 2]
    raz = rxl[:, 1]
    rel = rxl[:, 2]

    laz = np.kron(taz, np.ones(len(raz))).reshape(-1, len(raz)) + raz
    laz = laz.reshape(-1, 1)
    lel = np.kron(tel, np.ones(len(rel))).reshape(-1, len(rel)) + rel
    lel = lel.reshape(-1, 1)
    # Virtual antenna array steering matrix
    smat = np.exp(
        1j * np.pi * (laz * (np.cos(az) * np.sin(el)) + lel * np.cos(el))
    )
    return smat

def virtual_array(adc_samples: np.array,
                  txl,
                  rxl) -> np.array:
    """Generate the virtual antenna array matching the layout provided.

    Arguments:
        adc_samples: Raw ADC samples with the shape (ntx, nrx, nc, ns)
                        ntx: Number of TX antenna
                        nrx: Number of RX antenna
                        nc: Number of chirps per frame
                        ns: Number of samples per chirp
        txl: TX antenna layout array
                - Structure per row: [tx_idx, azimuth, elevation]
                - Unit: Half a wavelength
        rxl: RX antenna layout array
                - Structure: [tx_idx, azimuth, elevation]
                - Unit: Half a wavelength

    Return:
        The virtual antenna array of shape (nel, naz, nc, ns)
            nel: Number of elevation layers
            naz: Number of azimuth positions
            nc, ns: See above (description of `adc_samples`)

        See the variable `va_shape` to see how the shape is estimated
    """
    _, _, nc, ns = adc_samples.shape

    # Shape of the virtual antenna array
    va_shape: tuple = (
        # Length of the elevation axis
        # the "+1" is to count for the 0-indexing used
        np.max(txl[:, 2]) + np.max(rxl[:, 2]) + 1,

        # Length of the azimuth axis
        # the "+1" is to count for the 0-indexing used
        np.max(txl[:, 1]) + np.max(rxl[:, 1]) + 1,

        # Number of chirps per frame
        nc,

        # Number of samples per chirp
        ns,
    )

    # Virtual antenna array
    va = np.zeros(va_shape, dtype=np.complex128)

    # *idx: index of the antenna element
    # *az: azimuth of the antenna element
    # *el: elevation of the antenna element
    for tidx, taz, tel in txl:
        for ridx, raz, rel in rxl:
            # When a given azimuth and elevation position is already
            # populated, the new value is added to the previous to have
            # a strong signal feedback
            va[tel+rel, taz+raz, :, :] += adc_samples[tidx, ridx, :, :]
    return va


def fft_size(size: int) -> int:
    """Computed the closest power of 2 to be use for FFT computation.

    Argument:
        size: current size of the samples.

    Return:
        Adequate window size for FFT.
    """
    return 2 ** int(np.ceil(np.log(size) / np.log(2)))


def get_max_range(fs: float, fslope: float) -> float:
    """Compute the maximum range of the radar.

    Arguments:
        fs: Sampling frequency
        fslope: Chirp slope frequency
    """
    return fs * C / (2 * fslope)

def _get_bins(ns, nc, na, ne, radar_config): 
    """Return the range, velocity, azimuth and elevation bins.
    Arguments:
        ne: Elevation FFT size
        na: Azimuth FFT size
        nc: Doppler FFT size
        ns: Range FFT size

    Return:
        range bins
        velocity bins
        azimuth bins
        elevation bins

    NOTE: The bins are returned in the order listed above
    """
    # Number of TX antenna
    ntx: int = radar_config.numTxChan

    # ADC sampling frequency
    fs: float = radar_config.Fs

    # Frequency slope
    fslope: float = radar_config.Kr

    # Start frequency
    fstart: float = radar_config.StartFrequency

    # Ramp end time
    te: float = radar_config.chirpRampTime + radar_config.adc_start_time

    # Chirp time
    tc: float = radar_config.Ideltime + te

    rbins = np.array([])        # Range bins
    vbins = np.array([])        # Doppler bins
    abins = np.array([])        # Azimuth bins
    ebins = np.array([])        # Elevation bins

    AZIMUTH_FOV = np.deg2rad(radar_config.angles_DOA_az[1] - radar_config.angles_DOA_az[0])
    ELEVATION_FOV = np.deg2rad(radar_config.angles_DOA_ele[1] - radar_config.angles_DOA_ele[0])

    # Antenna PCB design base frequency
    fdesign: float = radar_config.F_design
    d = 0.5 * ((fstart / 1e9 + (fslope / 1e9 * radar_config.numAdcSamples / fs) / 2) / fdesign)
    if ns:
        rbins = get_range_bins(ns, fs, fslope)

    if nc:
        # Velocity bins
        vbins = get_velocity_bins(ntx, nc, fstart, tc)

    if na:
        # Azimuth bins
        ares = 2 * AZIMUTH_FOV / na
        # Estimate azimuth angles and flip the azimuth axis
        abins = -1 * np.arcsin(
            np.arange(-AZIMUTH_FOV, AZIMUTH_FOV, ares) / (
                2 * np.pi * d
            )
        )

    if ne:
        # Elevation
        eres = 2 * ELEVATION_FOV / ne
        # Estimate elevation angles and flip the elevation axis
        ebins = -1 * np.arcsin(
            np.arange(-ELEVATION_FOV, ELEVATION_FOV, eres) / (
                2 * np.pi * d
            )
        )
    return rbins, vbins, abins, ebins


def get_max_velocity(ntx:int, fstart: float, tc: float) -> float:
    """Compute the maximum range of the radar.

    Arguments:
        ntx: Number of TX antenna
        fstart: Chirp start frequency
        tc: Chirp time
    """
    return (C / fstart) / (4.0 * tc * ntx)


def get_range_resolution(ns: int, fs: float, fslope,
                        is_adc_filtered: bool = True) -> float:
    """Compute the range resolution of a Radar sensor.

    Arguments:
        ns: Number of ADC samples per chirp
        fs: Sampling frequency
        fslope: Chrip frequency slope
        is_adc_filtered: Boolean flag to indicate if a window function
        has been applied to the ADC data before processing. In such case
        the range resolution is affected. This parameter is set to True
        by default.

    Return:
        Range resolution in meter
    """
    rres: float = C / (ns * fslope / fs)
    if not is_adc_filtered:
        rres = rres / 2
    return rres


def get_velocity_resolution(nc: int, fstart: float, tc: float,
                            is_adc_filtered: bool = True) -> float:
    """Compute the vlocity resolution of a Radar sensor.

    Arguments:
        nc: Number of chirps per frame
        fstart: Start frequency of the chirp
        tc: Chirp time
            tc = Idle time + End time
        is_adc_filtered: Boolean flag to indicate if a window function
            has been applied to the ADC data before processing. In such case
            the velocity resolution is affected. This parameter is set to True
            by default.

    Return:
        Range velocity resolutio n in meter/s
    """
    lbd: float = C / fstart # lambda
    vres = lbd / (tc * nc)
    if not is_adc_filtered:
        vres = vres / 2
    return vres



def get_range_bins(ns: int, fs: float, fslope) -> np.array:
    """Return the range bins.

    Arguments:
        ns: Number of ADC samples per chirp
        fs: Sampling frequency
        fslope: Chrip frequency slope

    Return:
        Array of range bins
    """
    rmax: float = get_max_range(fs, fslope)
    # Resolution used for rendering
    # Note: Not the actual sensor resolution
    rres = rmax / ns
    return np.arange(0, rmax, rres)


def get_velocity_bins(ntx: int, nv: int, fstart: float, tc: float) -> np.array:
    """Compute the velocity bins

    Arguments:
        ntx:ntx: Number of transmission antenna
        nv: Number of expected velocity bins
        fstart: Start frequency of the chirp
        tc: Chirp time
            tc = Idle time + End time

    Return:
        Array of velocity bins
    """
    vmax: float = get_max_velocity(ntx, fstart, tc)
    # Resolution used for rendering
    # Not the actual radar resolution
    vres = (2 * vmax) / nv

    bins = np.arange(-vmax, vmax, vres)
    return bins


def os_cfar(samples: np.array, ws: int, ngc: int = 2, tos: int = 8,order_statistic_fraction: float = 0.75) -> np.array:  
    """Ordered Statistic Constant False Alarm Rate detector.

    Arguments:
        samples: Non-Coherently integrated samples
        ws: Window Size
        ngc: Number of guard cells
        tos: Scaling factor

    Return:
        mask
    """
    ns: int = len(samples)
    k: int = int(ws * order_statistic_fraction)

    # Add leading and trailing zeros into order to run the algorithm over
    # the entire samples
    samples = np.append(np.zeros(ws), samples)
    samples = np.append(samples, np.zeros(ws))

    mask = np.zeros(ns)
    for idx in range(ns):
        # tcells: training cells
        pre_tcells = samples[ws + idx - ngc - (ws // 2) : ws + idx - ngc]
        post_tcells = samples[ws + idx + ngc + 1: ws + idx + ngc + (ws // 2) + 1]
        tcells = np.array([])
        tcells = np.append(tcells, pre_tcells)
        tcells = np.append(tcells, post_tcells)
        tcells = np.sort(tcells)
        if samples[ws + idx] > tcells[k] * tos:
            mask[idx] = 1
    return mask


class ObjectDetected:
    """Object detected.

    Definition of object detected by applying CFAR

    NOTE: It's possible to have multiple detections on the same object
    depending on the resolution of the radar sensor
    """

    vidx: int = -1      # Velocity bin index
    ridx: int = -1      # Range bin index
    aidx: int = -1      # Azimuth bin index
    eidx: int = -1      # Elevation bin
    snr: float = 0      # Signal over Noise ratio

    def __str__(self) -> str:
        return f"Obj(SNR:{self.snr:.2f})"

    def __repr__(self) -> str:
        return self.__str__()


def nq_cfar_2d(samples, ws: int, ngc: int,
             quantile: float = 0.75, tos: int = 8) -> np.array:
    """N'th quantile statistic Constant False Alarm Rate detector.

    The principle is exactly the same as the Ordered Statistic
    Constant False Alarm Rate detector. This routine just applies
    it on a 2D signal.

    Arguments:
        samples: 2D signal to filter
        ws (int): Window size
        ngc (int): Number of guard cells
        quantile (float): Order of the quantile to compute for the noise
            power estimation
        tos (int): Scaling factor for detection an object
    """
    nx, ny = samples.shape
    mask = np.zeros((nx, ny))
    detections: list[ObjectDetected] = []

    for xidx in range(nx):
        # s:start index  e:end index  b:before  a:after
        # Before CUT (Cell Under Test) start index on the x-axis
        xbs: int = xidx - ws 
        xbs = xbs if (xbs > 0) else 0 

        # Before CUT (Cell Under Test) end index on the x-axis
        xbe: int = xidx - ngc
        xbe = xbe if (xbe > 0) else 0

        # After CUT (Cell Under Test) start index on the x-axis
        xas: int = xidx + ngc + 1
        # After CUT (Cell Under Test) end index on the x-axis
        xae: int =  xidx + ws + 1
        xae = xae if (xae < nx) else nx

        for yidx in range(ny): 
            # Before CUT (Cell Under Test) start index on the y-axis
            ybs: int = yidx - ws
            ybs = ybs if (ybs > 0) else 0

            # Before CUT (Cell Under Test) end index on the y-axis
            ybe: int = yidx - ngc

            # After CUT (Cell Under Test) start index on the y-axis
            yas: int = yidx + ngc + 1

            # After CUT (Cell Under Test) end index on the y-axis
            yae: int =  yidx + ws + 1
            yae = yae if (yae < ny) else ny

            tcells = np.array([]) 
            if xbe > 0:
                tcells = samples[xbs:xbe, ybs:yae].reshape(-1)  

            if xas < nx - 1:
                tcells = np.append( 
                    tcells,
                    samples[xas:xae, ybs:yae].reshape(-1)
                )

            if ybe > 0:
                tcells = np.append(
                    tcells,
                    samples[xbe:xas, ybs:ybe,].reshape(-1)
                )

            if yas < nx - 1:
                tcells = np.append(
                    tcells,
                    samples[xbe:xas, yas:yae,].reshape(-1)
                )
            m = np.quantile(tcells, quantile, method="weibull") 
            if samples[xidx, yidx] > (m * tos):
                mask[xidx, yidx] = 1
                obj = ObjectDetected()
                obj.vidx = xidx
                obj.ridx = yidx
                obj.snr = samples[xidx, yidx] / m
                detections.append(obj)
    return mask, detections



def mask_real_2d(cfar_power, radar_config , ws: int, ngc: int,
             quantile: float = 0.75, tos: int = 2 ) -> np.array:
    ne, na, nv, nr = cfar_power.shape                                       # (elevation, azimuth, doppler, range)
    mask = np.zeros((ne, na, nr))                                           # (elevation, azimuth, range)
    dopplar_real = np.zeros((ne, na, nr))                                   # (elevation, azimuth, range)
    rbins, vbins, abins, ebins = _get_bins(nr, nv, na, ne, radar_config)
    doppler_max_indices = np.argmax(cfar_power, axis=2)                     #  (elevation, azimuth, range)
    for e in range(ne):
        for a in range(na):
            for r in range(nr):
                v = doppler_max_indices[e, a, r]
                # Before CUT (Cell Under Test) start index on the v-axis                
                vbs: int = v - ws  
                vbs = vbs if (vbs > 0) else 0  

                # Before CUT (Cell Under Test) end index on the v-axis
                vbe: int = v - ngc
                vbe = vbe if (vbe > 0) else 0

                # After CUT (Cell Under Test) start index on the v-axis
                vas: int = v + ngc + 1
                vas = vas if (vas < nv) else nv    

                # After CUT (Cell Under Test) end index on the v-axis
                vae: int = v + ws + 1
                vae = vae if (vae < nv) else nv    

                # Before CUT (Cell Under Test) start index on the r-axis
                rbs: int = r - ws
                rbs = rbs if (rbs > 0) else 0  

                # Before CUT (Cell Under Test) end index on the r-axis
                rbe: int = r - ngc
                rbe = rbe if (rbe > 0) else 0  

                # After CUT (Cell Under Test) start index on the r-axis
                ras: int = r + ngc + 1
                ras = ras if (ras < nr) else nr

                # After CUT (Cell Under Test) end index on the r-axis
                rae: int = r + ws + 1
                rae = rae if (rae < nr) else nr

                tcells = np.array([]) 

                # Before doppler dimension area of CUT
                if vbe > 0:
                    tcells = cfar_power[e, a, vbs:vbe, rbs:rae].reshape(-1)

                # After doppler dimension area of CUT
                if vas < nv - 1:  
                    tcells = np.append(
                        tcells,
                        cfar_power[e, a, vas:vae, rbs:rae].reshape(-1)
                    )

                # Before range dimension area of CUT
                if rbe > 0:
                    tcells = np.append(
                        tcells,
                        cfar_power[e, a, vbe:vas, rbs:rbe].reshape(-1)
                    )

                # After range dimension area of CUT
                if ras < nr - 1:
                    tcells = np.append(
                        tcells,
                        cfar_power[e, a, vbe:vas, ras:rae].reshape(-1)
                    )         
                m = np.quantile(tcells, quantile, method="weibull") 
                if cfar_power[e, a, v, r] > (m * tos):
                    mask[e, a ,r] = 1
                    dopplar_real[e, a ,r] = vbins[v]
                else:
                    mask[e, a ,r] = 0
                    dopplar_real[e, a ,r] = 0
    return mask, dopplar_real


def velocity_compensation( ntx, nc) -> None:
    """Generate the compensation matrix for velocity-induced phase-shift.

    Meant to compensate the velocity-induced phase shift created by TDM MIMO
    configuration.

    Arguments:
        ntx: Number of transmission antenna
        nc: Number of chirps per frame

    Return:
        Phase shift correction matrix
    """
    tl = np.arange(0, ntx)
    cl = np.arange(-nc//2, nc//2)
    tcl = np.kron(tl, cl) / (ntx * nc)
    # Velocity compensation
    vcomp = np.exp(-2j * np.pi * tcl)
    vcomp = vcomp.reshape(ntx, 1, nc, 1)
    return vcomp

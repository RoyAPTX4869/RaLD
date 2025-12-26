import numpy as np
import dataset_preprocessor.utils.radardsp as radardsp

NOISE_THRESHOLD = 0.30  # 30th percentile

def RAEIVVmap(radar_adc_data,radar_config, tx_array, rx_array):
    ''' 
        @Author: WSP
        Args:
            radar_adc_data: (ntx, nrx, nc, ns)
            radar_config: radar configuration
            tx_array: (ntx, 3)
            rx_array: (nrx, 3)
        Returns:
            RAEmap: (range, azimuth, elevation, 3)
                0: intensity
                1: velocity
                2: validity
    '''
    ntx, nrx, nc, ns = radar_adc_data.shape

    radar_adc_data *= np.blackman(ns).reshape(1, 1, 1, -1)

    rfft = np.fft.fft(radar_adc_data, radar_config.range_fftsize, -1)    

    dfft = np.fft.fft(rfft, radar_config.doppler_fftsize, -2)
    dfft = np.fft.fftshift(dfft, -2)
    vcomp = radardsp.velocity_compensation(ntx, radar_config.doppler_fftsize) 
    dfft *= vcomp
    
    _dfft = radardsp.virtual_array(dfft, tx_array, rx_array) 

    afft = np.fft.fft(_dfft, radar_config.ANGLE_fftsize, 1) 
    # Spectral shift
    afft = np.fft.fftshift(afft, 1) 


    efft = np.fft.fft(afft, radar_config.ELEVATION_fftsize, 0) 
    efft = np.fft.fftshift(efft, 0)

    # Set the values of the head and tail parts to 0
    efft[:, :, :, 0:int(efft.shape[-1] * radar_config.crop_low)] = 0    
    efft[:, :, :, -int(efft.shape[-1] * radar_config.crop_high):] = 0
    ne, na, nv, nr = efft.shape  
    _, vbins, _, _ = radardsp._get_bins(nv, nr, na, ne, radar_config)   
    FFT_power = np.abs(efft) ** 2                                                         # (elevation, azimuth, doppler, range)

    max_indices = np.argmax(FFT_power, axis=2)                                            # (elevation, azimuth, range)
    max_velocity = vbins[max_indices]                                                     # (elevation, azimuth, range)
    max_velocity = np.transpose(max_velocity, (2, 1, 0))                                  # (range,azimuth,elevation) Velocity
    sorted_values = np.sort(FFT_power, axis=2)
    valid_mask = sorted_values[:, :, -1] * (1-NOISE_THRESHOLD) > sorted_values[:, :, -2]
    valid_mask = np.transpose(valid_mask, (2, 1, 0))                                      # (range,azimuth,elevation) Mask
    FFT_power = np.sum(FFT_power, axis=-2)                                                # (elevation, azimuth, range)
    noise = np.quantile(FFT_power, NOISE_THRESHOLD, (0,1,2))                              # (elevation, azimuth)
    FFT_power /= (noise+1e-6)
    # Convert to dB
    dpcl = 10 * np.log10(FFT_power + 1)  
    dpcl_trans = np.transpose(dpcl, (2, 1, 0))                                            # (range,azimuth,elevation) Intensity
    # concat intensity, velocity, validity
    result = np.stack((dpcl_trans, max_velocity, valid_mask), axis=-1, dtype=np.float32)  # (range,azimuth,elevation,3)
    return result
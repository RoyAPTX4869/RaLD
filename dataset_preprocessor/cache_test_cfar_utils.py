import numpy as np
from skimage.feature import peak_local_max
import torch
import numpy as np
from torch.nn import functional as F
from constants import WAVELENGTH_TO_APERTURE_RATIO


def read_radar_map_bin(radar_path, config):
    '''
        Args:
            path: str, radar point cloud file path
        Returns:
            radar_map: (R,A,E,2)
    '''
    try:
        radar_cube = np.fromfile(radar_path, dtype=np.float32)
    except Exception as e:
        print(f"Error loading radar file {radar_path}")
        print(e)
        raise ValueError(f"Error loading radar file {radar_path}")
    radar_cube = radar_cube.reshape(config.input_r_size, config.input_a_size, \
                                    config.input_e_size, -1)
    return torch.from_numpy(radar_cube[:,:,:,:2])  # return Tensor (R, A, E, 2)

def rae_interpo(rae_map, R, A, E):
    """
    Interpolate rae_map to the specified dimensions (B, R, A, E)

    Args:
        rae_map (torch.Tensor): Input tensor of shape (B, R_in, A_in, E_in)
        R, A, E (int): Target dimensions

    Returns:
        torch.Tensor: Interpolated tensor of shape (B, R, A, E) 
    """
    # Adjust dimensions to match F.interpolate expected input (B, C, D, H, W)
    # Adjust to: (B, C=1, D=R_in, H=A_in, W=E_in)
    rae_map = rae_map.unsqueeze(1)  # (B, 1, 128, 8, 2)

    # Use trilinear interpolation to resize
    # Note: mode='trilinear' requires a 5D tensor input (B, C, D, H, W)
    rae_map_ = F.interpolate(
        rae_map, 
        size=(R, A, E), 
        mode='trilinear', 
        align_corners=False
    )  # (B, 1, R, A, E)

    # Remove the temporarily added channel dimension
    rae_map_ = rae_map_.squeeze(1)  # (B, R, A, E)

    return rae_map_

def RA2DDetector(ramap_cube, num=1000):

    """
    Perform 2D peak detection on each slice of a 3D data cube.

    Args:
    ramap_cube (np.ndarray): 3D array of shape (R, A, D)
    num: Number of peaks to detect per slice. 
    Returns:
    list: List of peak coordinates for each slice, where each element is an (N, 2) array of peak coordinates in that slice
    """
    all_peaks = []
    all_intensities = []

    # Process each 2D slice
    for r in range(ramap_cube.shape[0]):
        if not num[r]:
            continue
        ramap_2d = ramap_cube[r, :, :]

        assert num[r]<=ramap_2d.shape[0]*ramap_2d.shape[1]
        flat_indices = np.argpartition(ramap_2d.flatten(), -num[r])[-num[r]:]
        flat_indices = flat_indices[np.argsort(-ramap_2d.flatten()[flat_indices])]  # sorted by intensity

        # Change flat indices to 2D coordinates
        rows = flat_indices // ramap_2d.shape[1]  
        cols = flat_indices % ramap_2d.shape[1]   
        peaks = np.column_stack((rows, cols))
        intensities = ramap_2d[rows, cols]

        all_peaks.append(np.stack((
                                   np.ones_like(peaks[:, 0]) * r,  # r
                                   peaks[:, 0],                    # a
                                   peaks[:, 1]), axis=-1))         # e (N, 3) 
        all_intensities.append(intensities)                        # list (N,)

        assert len(intensities) == num[r]
    if all_peaks:   # Ensure the list is not empty
        all_peaks = np.concatenate(all_peaks, axis=0)   # (N, 3)
        all_intensities = np.concatenate(all_intensities, axis=0)
    else:
        assert "all_peaks is None! "

    return all_peaks, all_intensities

def weighted_allocation(weights, total):
    """
    Weight allocation implemented with PyTorch tensors: proportional allocation of integers, with the maximum weight item bearing any surplus/deficit.

    Args:
        weights: Weight tensor (torch.Tensor, non-negative, shape (N,))
        total: Total number to allocate (integer)
    
    Returns:
        torch.Tensor: Allocation result (integer tensor, shape (N,), sum equals total)
    """

    # Ensure the input is a tensor and of the correct type
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights, dtype=torch.float32)
    else:
        weights = weights.to(dtype=torch.float32)

    n = weights.numel()
    total_weight = weights.sum()

    # When all weights are zero, allocate evenly
    if total_weight == 0:
        base = total // n
        remain = total % n
        alloc = torch.full((n,), base, dtype=torch.int64, device=weights.device)
        if remain > 0:
            alloc[:remain] += 1
        return alloc

    ratios = weights / total_weight
    alloc = (ratios * total).floor().to(torch.int64)
    current_sum = alloc.sum()
    diff = total - current_sum  

    # find the index of the maximum weight (if multiple maxima, take the first)
    max_idx = torch.argmax(weights)

    # Adjust the allocation for the maximum weight item
    alloc[max_idx] += diff

    return alloc

def RA2DDetectorTensor(ramap_cube, num=10000):
    '''
        Args: 
            ramap_cube: (B, R, A, E)
        Returns:
            all_peaks: (B, N, 3)
            all_intensities: (B, N)
    '''

    all_peaks = torch.zeros((ramap_cube.shape[0], num, 3))
    all_intensities = torch.zeros((ramap_cube.shape[0], num))
    for b in range(ramap_cube.shape[0]):
        mini_num = weighted_allocation(ramap_cube[b].sum(axis=[1,2])/ramap_cube[b].sum(), num)

        peaks, intensity = RA2DDetector(ramap_cube[b].numpy(), num=mini_num)
        all_peaks[b] = torch.from_numpy(peaks)
        all_intensities[b] = torch.from_numpy(intensity)
    return all_peaks.squeeze(0).int(), all_intensities.squeeze(0)

def cube_idx2coord(idx, config, return_in_degrees=False):
    '''
    Arguments:
        idx: (N, 3) numpy array, where each row is (r, a, e) indices
        config: configuration object containing radar parameters
    Returns:
        coords: (N, 3) numpy array, where each row is (r, a, e) coordinates
    Description:
        Converts indices in the radar cube to coordinates in the (r, a, e) space
    '''

    assert idx.shape[1] == 3, "Index should have shape (N, 3) for (r, a, e)"

    # range_cell_size = 0.1004*2
    r_size = config.target_r_size
    max_range = config.max_range
    range_cell_size = max_range / r_size  # Calculate range cell size based on max range and number of range cells
    # Range Axis
    az_size = config.target_a_size
    el_size = config.target_e_size


    range_axis = np.arange(range_cell_size, max_range + range_cell_size / 2, range_cell_size)
    # Azimuth Axis
    wx_vec = np.linspace(-np.pi, np.pi, az_size)
    wx_vec = np.flip(wx_vec) 
    azimuth_axis = np.arcsin(np.clip(wx_vec / (2 * np.pi * WAVELENGTH_TO_APERTURE_RATIO), -1, 1)) # 
    azimuth_axis[0] = np.pi/2
    azimuth_axis[-1] = -np.pi/2
    # Elevation Axis
    wz_vec = np.linspace(-np.pi, np.pi, el_size)
    wz_vec = np.flip(wz_vec)
    elevation_axis = np.arcsin(np.clip(wz_vec / (2 * np.pi * WAVELENGTH_TO_APERTURE_RATIO), -1, 1))  # Clip to avoid NaN values due to arcsin domain issues
    elevation_axis[0] = np.pi/2
    elevation_axis[-1] = -np.pi/2
    elevation_axis=-elevation_axis
    azimuth_axis=-azimuth_axis

    if return_in_degrees:
        azimuth_axis = np.rad2deg(azimuth_axis)
        elevation_axis = np.rad2deg(elevation_axis)

    coords = np.zeros_like(idx, dtype=np.float32)
    coords[:, 0] = range_axis[idx[:, 0].numpy()]  # Range coordinates
    coords[:, 1] = azimuth_axis[idx[:, 1].numpy()]  # Azimuth
    coords[:, 2] = elevation_axis[idx[:, 2].numpy()]  # Elevation
    return coords


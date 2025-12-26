import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

def filter_timestamps(timestamps: np.array, start_time: float, end_time: float):
    """Filter timestamps by start and end time.

    args:
        timestamps: Timestamps in the format of [N]
        start_time: Start time
        end_time: End time
    return:
        valid_mask: Mask of valid timestamps
        filtered_timestamps: Filtered timestamps
    """
    valid_mask = (timestamps >= start_time) & (timestamps <= end_time)
    filtered_timestamps = timestamps[valid_mask]
    return valid_mask, filtered_timestamps

def interp_pose(gt_timestamps: np.array, gt_poses: np.array, target_timestamps: np.array):
    '''
    This function interpolates the ground truth poses to the target timestamps.
    
    args:
        gt_timestamps: timestamps of the ground truth poses in the format of [N]
        gt_poses: ground truth poses in the format of [N, 4, 4]
        target_timestamps: target timestamps in the format of [M]
    return:
        interp_poses: interpolated poses in the format of [M 4, 4]
    '''
    # check if the target timestamps are within the range of the ground truth timestamps
    if target_timestamps[0] < gt_timestamps[0] or target_timestamps[-1] > gt_timestamps[-1]:
        raise ValueError("Target timestamps are not within the range of the ground truth timestamps.")
    gt_rot = R.from_matrix(gt_poses[:, :3, :3])
    gt_trans = gt_poses[:, :3, 3]
    
    # interpolate the rotation
    slerp = Slerp(gt_timestamps, gt_rot)
    interp_rot = slerp(target_timestamps)
    interp_rot = interp_rot.as_matrix()
    
    # interpolate the translation
    interp_trans = np.zeros((len(target_timestamps), 3))
    for i in range(3):
        interp_trans[:, i] = interp1d(gt_timestamps, gt_trans[:, i])(target_timestamps)
    
    interp_poses = np.zeros((len(target_timestamps), 4, 4))
    interp_poses[:, :3, :3] = interp_rot
    interp_poses[:, :3, -1] = interp_trans
    return interp_poses
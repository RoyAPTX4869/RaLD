import numpy as np
import torch
from pathlib import Path
from os import PathLike
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


def get_inverse_tf(T):
    """Returns the inverse of a given 4x4 homogeneous transform.
    Args:
        T (np.ndarray): 4x4 transformation matrix
    Returns:
        np.ndarray: inv(T)
    """
    T2 = np.identity(4, dtype=np.float32)
    R = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)
    T2[0:3, 0:3] = R.transpose()
    T2[0:3, 3:] = np.matmul(-1 * R.transpose(), t)
    return T2

def get_inverse_tf_torch(T):
    """Returns the inverse of a given 4x4 homogeneous transform.
    Args:
        T tensor. 4x4 transformation matrix
    Returns:
        torch.tensor: inv(T)
    """
    T2 = torch.eye(4)
    R = T[0:3, 0:3]
    t = T[0:3, 3].reshape(3, 1)
    T2[0:3, 0:3] = R.T
    T2[0:3, 3:] = torch.mm(-1 * R.T, t)
    return T2

def ensure_path_exists(path: PathLike) -> bool:
	"""
	ensure path exists
	
	return:
		True if path exists, otherwise False
	"""
	if not Path(path).exists(): 
		Path(path).mkdir(parents=True, exist_ok=True)
		return False
	else:
		return True

def inverse_norm_points(points, lidar_pc_range, norm_anisotropy, norm_isotropy):
    """
    inverse normalization
    Args:
        points: normalized points, numpy array of shape (N, 3)
        lidar_pc_range: list of 6 elements, [x_min, y_min, z_min, x_max, y_max, z_max]
        norm_anisotropy: bool, whether to use anisotropic normalization
        norm_isotropy: bool, whether to use isotropic normalization
    Returns:
        pred: inverse normalized points, numpy array of shape (N, 3)
    """
    x_offset = (lidar_pc_range[3] + lidar_pc_range[0]) / 2
    y_offset = (lidar_pc_range[4] + lidar_pc_range[1]) / 2
    z_offset = (lidar_pc_range[5] + lidar_pc_range[2]) / 2
    x_scale = (lidar_pc_range[3] - lidar_pc_range[0]) / 2
    y_scale = (lidar_pc_range[4] - lidar_pc_range[1]) / 2
    z_scale = (lidar_pc_range[5] - lidar_pc_range[2]) / 2
    pred = np.zeros_like(points)
    if norm_anisotropy:
        pred[:, 0] = points[:, 0] * x_scale + x_offset
        pred[:, 1] = points[:, 1] * y_scale + y_offset
        pred[:, 2] = points[:, 2] * z_scale + z_offset
    if norm_isotropy:
        max_scale = max(x_scale, y_scale, z_scale)
        offset = np.array([x_offset, y_offset, z_offset])
        pred[:, :3] = points[:, :3] * max_scale + offset
    return pred

def norm_points(points, lidar_pc_range, norm_anisotropy, norm_isotropy):
    """
    normalization
    Args:
        points: points to be normalized, numpy array of shape (N, 3)
        lidar_pc_range: list of 6 elements, [x_min, y_min, z_min, x_max, y_max, z_max]
        norm_anisotropy: bool, whether to use anisotropic normalization
        norm_isotropy: bool, whether to use isotropic normalization
    Returns:
        normed_pred: normalized points, numpy array of shape (N, 3)
    """
    x_offset = (lidar_pc_range[3] + lidar_pc_range[0]) / 2
    y_offset = (lidar_pc_range[4] + lidar_pc_range[1]) / 2
    z_offset = (lidar_pc_range[5] + lidar_pc_range[2]) / 2
    x_scale = (lidar_pc_range[3] - lidar_pc_range[0]) / 2
    y_scale = (lidar_pc_range[4] - lidar_pc_range[1]) / 2
    z_scale = (lidar_pc_range[5] - lidar_pc_range[2]) / 2
    normed_pred = np.zeros_like(points)
    if norm_anisotropy:
        normed_pred[:, 0] = (points[:, 0] - x_offset) / x_scale
        normed_pred[:, 1] = (points[:, 1] - y_offset) / y_scale
        normed_pred[:, 2] = (points[:, 2] - z_offset) / z_scale
    if norm_isotropy:
        max_scale = max(x_scale, y_scale, z_scale)
        offset = np.array([x_offset, y_offset, z_offset])
        normed_pred[:, :3] = (points[:, :3] - offset) / max_scale
    return normed_pred

def remove_points_outside_fov(points):
    # create a boolean mask to keep points with all coordinates strictly within (-1, 1)
    mask = np.all((points > -1) & (points < 1), axis=1)

    # Filter data using the mask
    filtered_points = points[mask]
    return filtered_points


################# Accelerating metric calculating by KD-Tree #########
def cal_metrics(y_pred, y_gt):
    if len(y_pred)==0:
         return np.inf

    pred_tree = cKDTree(y_pred)
    gt_tree = cKDTree(y_gt)

    min_dist_pred_to_gt = []
    min_dist_gt_to_pred = []
    # get the min_dist from pred to gt
    for i in range(len(y_pred)):
        dist, _ = gt_tree.query(y_pred[i])
        min_dist_pred_to_gt.append(dist)
    for i in range(len(y_gt)):
        dist, _ = pred_tree.query(y_gt[i])
        min_dist_gt_to_pred.append(dist)

    # dist_matrix = cdist(y_pred, y_gt, 'euclidean')
    cd = chamfer_distance(min_dist_gt_to_pred=min_dist_gt_to_pred, 
                             min_dist_pred_to_gt=min_dist_pred_to_gt)

    return cd

def chamfer_distance(min_dist_pred_to_gt, min_dist_gt_to_pred):
    chamfer_gt_to_pred = np.mean(min_dist_gt_to_pred)
    chamfer_pred_to_gt = np.mean(min_dist_pred_to_gt) 
    return 0.5 * chamfer_gt_to_pred + 0.5 * chamfer_pred_to_gt

################# Accelerating metric calculating by KD-Tree #########


def generate_query_points(args,coordinate_type='polar'):
    # generate query points within the normalized space
    num_points = args.eval.inference.num_query_points
    if coordinate_type=='polar':
        pc_range = args.dataset.lidar.pc_range
    elif coordinate_type=='cart':
        pc_range = args.dataset.lidar.pc_range_cart
    else:
        raise ValueError("coordinate_type must be 'polar' or 'cart'")
    x_scale = (pc_range[3] - pc_range[0]) / 2
    y_scale = (pc_range[4] - pc_range[1]) / 2
    z_scale = (pc_range[5] - pc_range[2]) / 2
    max_scale = max(x_scale, y_scale, z_scale)
    if args.dataset.lidar.norm_anisotropy:
        x_min,y_min,z_min = -1,-1,-1
        x_max,y_max,z_max =  1, 1, 1

    if args.dataset.lidar.norm_isotropy:
        x_min = -(x_scale/max_scale)
        x_max = x_scale/max_scale
        y_min = -(y_scale/max_scale)
        y_max = y_scale/max_scale
        z_min = -(z_scale/max_scale)
        z_max = z_scale/max_scale
    x = np.random.uniform(x_min, x_max, num_points)
    y = np.random.uniform(y_min, y_max, num_points)
    z = np.random.uniform(z_min, z_max, num_points)

    grid_np = np.stack([x, y, z], axis=1)
    return grid_np
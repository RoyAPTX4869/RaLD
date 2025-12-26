import numpy as np

def aug_query_helper(helper_points, aug_num, pc_range, voxel_size, aug_bias_scale=2):
    '''
    Apply augmentations to helper point. Given a set of helper points, 
    adding augmented points by applying random bias to the original points.
    The bias is scaled by a factor of `aug_bias_scale`, which means the bias of each 
    augmented points is within aug_bias_scale*voxel size.
    The augmented points should be guaranteed to be within the pc range.

    Args:
        helper_points: np.array, in shape of [N,3]
        aug_num: int
        pc_range: np.array, [x_min, y_min, z_min, x_max, y_max, z_max]
        voxel_size: np.array, [x,y,z]
        aug_bias_scale: int

    Return:
        aug_helper_points: np.array, in shape pf [aug_num, 3]
    '''
    assert helper_points.shape[1] == 3

    N = helper_points.shape[0]
    aug_helper_points = np.zeros((aug_num,3), np.float32)

    if N >= aug_num:
        aug_helper_points[:aug_num, :] = helper_points[:aug_num, :]
        return aug_helper_points

    generated_num = aug_num - N
    selected_points_idx = np.random.choice(N, size=generated_num, replace=True)
    possible_aug_s = np.arange(aug_bias_scale, step=1) + 1

    # Vectorized bias generation
    points = helper_points[selected_points_idx]  # shape: [generated_num, 3]
    aug_scales = np.random.choice(possible_aug_s, size=generated_num)  # shape: [generated_num]
    biases = (np.random.rand(generated_num, 3) * 2 - 1) * (voxel_size * aug_scales[:, None])  # shape: [generated_num, 3]
    aug_points = points + biases
    aug_points = np.clip(aug_points, pc_range[:3], pc_range[3:])

    aug_helper_points[:N, :] = helper_points
    aug_helper_points[N:, :] = aug_points
    return aug_helper_points
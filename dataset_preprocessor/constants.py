from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import numpy as np

EXCLUDE_DIR_NAMES = ['calib']

## hard code the transformation matrix, define in calib/base_to_lidar.txt and calib/base_to_single_chip.txt
BASED_TO_RADAR = {
    "translation": [-0.145, 0.09, -0.025],
    "quaternion": [0.0, 0.0, 0.706825181105, 0.707388269167]
}

BASED_TO_LIDAR = {
    "translation": [-0.075, -0.02, 0.03618],
    "quaternion": [0.0, 0.0, 0.721382357437, -0.692536998563]
}

WAVELENGTH_TO_APERTURE_RATIO = 0.4972  # degrees
T_BASED_TO_RADAR = np.eye(4)
T_BASED_TO_RADAR[:3, :3] = Rotation.from_quat(BASED_TO_RADAR["quaternion"]).as_matrix()
T_BASED_TO_RADAR[:3, 3] = BASED_TO_RADAR["translation"]

T_BASED_TO_LIDAR = np.eye(4)
T_BASED_TO_LIDAR[:3, :3] = Rotation.from_quat(BASED_TO_LIDAR["quaternion"]).as_matrix()
T_BASED_TO_LIDAR[:3, 3] = BASED_TO_LIDAR["translation"]

T_RADAR_TO_LIDAR = np.linalg.inv(T_BASED_TO_RADAR) @ T_BASED_TO_LIDAR
# print('Translation:', T_RADAR_TO_LIDAR[:3, 3])
# print("Rotation in Euler angles (xyz):", Rotation.from_matrix(T_RADAR_TO_LIDAR[:3, :3]).as_euler('xyz', degrees=True))


# The recorded attributes of lidar are:
# x, y, z, I (Intensity of the reflections)
NUMBER_RECORDING_ATTRIBUTES = 4

############# constants for hustRadar dataset #############
HUST_T_LIDAR_TO_SINGLE_CHIP = np.eye(4)
x_shift = 41.7
y_shift = 6.6
x = 25.2 + 62.74 - 8
y = 230 - (24.6 + x_shift)
z = -(65.5 + y_shift) - 23
# from scipy.spatial.transform import Rotation
# R = Rotation.from_euler('z', np.pi/2, degrees=False).as_matrix()
R = np.eye(3)
HUST_T_LIDAR_TO_SINGLE_CHIP[:3, :3] = R
HUST_T_LIDAR_TO_SINGLE_CHIP[:3, 3] = np.array([x, y, z])/1000
HUST_T_SINGLE_CHIP_TO_LIDAR = np.linalg.inv(HUST_T_LIDAR_TO_SINGLE_CHIP)
HUST_DIR_NAMES = [
    "classroom_f404_20241219_0",
    "classroom_f404_20241219_1",
    "classroom_f404_20241219_2"
]
HUST_NUMBER_RECORDING_ATTRIBUTES = 4
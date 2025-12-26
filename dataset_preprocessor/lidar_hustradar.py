'''
This script is used to preprocess the lidar data in the Hustradar dataset with the following steps:
    1. Load the lidar data
    2. Transform the lidar data from lidar coord to radar coord
    3. Select the overlapping time stamps of radar and lidar
    3. Crop the lidar data based on the field of view (FOV) of the radar
    4. Save the preprocessed lidar data
'''


import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import yaml
from easydict import EasyDict

from dataset_preprocessor.constants import HUST_T_LIDAR_TO_SINGLE_CHIP, HUST_NUMBER_RECORDING_ATTRIBUTES, HUST_DIR_NAMES
from utils.utils import ensure_path_exists

def arg_parser():
    parser = argparse.ArgumentParser(description="Convert Radar data to Lidar data")
    parser.add_argument("--config", type=str, 
                        default="dataset_preprocessor/config/lidar_config_hustradar.yaml",)
    parser.add_argument("--mode", type=str, 
                        default="sc", choices=["sc", "cc"],
                        help="Mode to convert data, sc: single-chip radar, cs: cascade radar")
    return parser.parse_args()

def load_lidar_data(lidar_path:Path, return_xyz=True)->np.array:
    try:
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, HUST_NUMBER_RECORDING_ATTRIBUTES)
    except:
        raise IOError
    if return_xyz:
        return points[:, :3]
    else:
        return points

def transform_lidar_data(points:np.array) -> np.array:
    assert points.shape[1] == 3
    points = np.hstack([points, np.ones((points.shape[0], 1))])
    points = points @ HUST_T_LIDAR_TO_SINGLE_CHIP
    return points[:, :3]

def cartesian2polar(points:np.array) -> np.array:
    assert points.shape[1] == 3
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    az = - np.rad2deg(np.arctan2(y, x))
    el = np.rad2deg(np.arcsin(z / r))
    return np.stack([r, az, el], axis=1)

def polar2cartesian(points:np.array) -> np.array:
    assert points.shape[1] == 3
    r, az, el = points[:, 0], - np.deg2rad(points[:, 1]), np.deg2rad(points[:, 2])
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return np.stack([x, y, z], axis=1)

def save_lidar_data(points:np.array, save_path:Path)->None:
    # save as binary file
    points = points.astype(np.float32)
    points.tofile(save_path)
    return

def filter_points_polar(points:np.array, range:list[np.array])->np.array:
    ''' Filter points based on polar coordinates
    Args:
        points: numpy array of shape (N, 3), in polar coordinates
        range: list of numpy arrays of shape (3, ) representing the range of:
            - r: radius, in meters
            - az: azimuth, in degrees
            - el: elevation, in degrees
    Returns:
        numpy array of shape (M, 3)
    '''
    assert points.shape[1] == 3, "Input points must be in polar coordinates"
    mask = np.logical_and.reduce(
        [points[:, 0] >= range[0][0], points[:, 0] <= range[0][1],
         points[:, 1] >= range[1][0], points[:, 1] <= range[1][1],
         points[:, 2] >= range[2][0], points[:, 2] <= range[2][1]]
    )
    return points[mask]

def remove_empty_points(points:np.array)->np.array:
    ''' Remove points with all zero coordinates, in cartesian coordinates
    Coloradar data set has some points with all zero coordinates, which are not valid
    Args:
        points: numpy array of shape (N, 3), in cartesian coordinates
    Returns:
        numpy array of shape (M, 3)
    '''
    xyz = points[:, :3]
    mask = np.linalg.norm(xyz, axis=1) > 0
    return points[mask]

def main():
    args = arg_parser()
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, yaml.FullLoader)
    config = EasyDict(config)
    print(config)

    dataset_dir = Path(config.root_dir)
    out_base_dir = Path(config.output_dir)
    sequence_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name in HUST_DIR_NAMES]
    print(f"Found {len(sequence_dirs)} sequences in {dataset_dir}")

    for seq_dir in tqdm(sequence_dirs):
        lidar_dir = seq_dir / "lidar"
        lidar_files = list(lidar_dir.glob("*.bin"))
        lidar_files.sort(key=lambda x: float(x.stem.split("_")[-1]))
        out_dir = out_base_dir / seq_dir.name / "lidar_sc"
        ensure_path_exists(out_dir)
        print(f"Found {len(lidar_files)} lidar files in {seq_dir.name}")

        # # load time stamps of lidar and radar
        # lidar_time_stamps = Path(seq_dir / "timestamps.txt")
        # with open(lidar_time_stamps, 'r') as f:
        #     lidar_time_stamps = f.readlines()

        index_file = Path(seq_dir / "lidar_index_sequence.txt")
        with open(index_file, 'r') as f:
            lindex = f.readlines()
        lindex = [int(i) for i in lindex]
        print(f"Found {len(lindex)} overlapping time stamps of lidar and radar")

        for i,index in tqdm(enumerate(lindex)):
            lidar_file = lidar_files[index]
            lidar_points = load_lidar_data(lidar_file)
            lidar_points = remove_empty_points(lidar_points)
            lidar_points = transform_lidar_data(lidar_points)
            polar_lidar_points = cartesian2polar(lidar_points)

            fov = config.single_chip_mode.lidar.FOV
            range_limits = [
                [0, fov.max_range], # range limits always start from 0
                [fov.az_range[0], fov.az_range[1]],
                [fov.el_range[0], fov.el_range[1]]
            ]
            filtered_polar_lidar_points = filter_points_polar(polar_lidar_points, range_limits)
            filtered_lidar_points = polar2cartesian(filtered_polar_lidar_points)
            out_dir_i = out_dir / f"{i:04d}.bin"
            save_lidar_data(filtered_lidar_points, out_dir_i)

    return


if __name__ == "__main__":
    main()
    
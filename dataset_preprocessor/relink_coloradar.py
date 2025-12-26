import os
from pathlib import Path
import numpy as np

base_dir_names = ['cascade', 'groundtruth', 'imu', 'lidar', 'single_chip']
exclude_dir_names = ['ColoRadar_tools-master', 'zip']
special_dir_names = ['calib']

def create_baselink(src_dir, dst_dir):
    seq_dirs = [d for d in src_dir.iterdir() if d.is_dir() and d.name not in exclude_dir_names and d.name not in special_dir_names]
    print(f"Found {len(seq_dirs)} sequences in {src_dir}")
    for seq_dir in seq_dirs:
        dst_seq_dir = dst_dir / seq_dir.name
        dst_seq_dir.mkdir(parents=True, exist_ok=True)
        for base_dir_name in base_dir_names:
            src_base_dir = seq_dir / base_dir_name
            dst_base_dir = dst_seq_dir / base_dir_name
            dst_base_dir.symlink_to(src_base_dir, target_is_directory=True)
            print(f"Created symlink: {dst_base_dir} -> {src_base_dir}")
    # link special dirs
    for special_dir_name in special_dir_names:
        src_special_dir = src_dir / special_dir_name
        dst_special_dir = dst_dir / special_dir_name
        dst_special_dir.symlink_to(src_special_dir, target_is_directory=True)
        print(f"Created symlink: {dst_special_dir} -> {src_special_dir}")

def main():
    src_dir = Path('/storage/public_dataset/coloradar')
    dst_dir = Path('/data/ruijiezzz/dataset/coloradar')
    create_baselink(src_dir, dst_dir)

if __name__ == "__main__":
    main()
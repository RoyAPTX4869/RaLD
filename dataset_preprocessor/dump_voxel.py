import os 
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import yaml
from easydict import EasyDict

from datasets.utils.voxelize import VoxelGeneratorWrapper
from utils.utils import ensure_path_exists
from utils.concurrent import imap_tqdm
from dataset_preprocessor.lidar import cartesian2polar

def arg_parser():
    parser = argparse.ArgumentParser(description="Convert Radar data to Lidar data")
    parser.add_argument("--config", type=str, 
                        default="dataset_preprocessor/config/coloradar_config.yaml",)
    parser.add_argument("--mode", type=str, 
                        default="sc", choices=["sc", "cc", "sc_cone"],
                        help="Mode to convert data, sc: single-chip radar, cs: cascade radar")
    return parser.parse_args()

def load_lidar(lidar_path):
    try:
        lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 3)
    except Exception as e:
        print(f"Error loading lidar file {lidar_path}")
        print(e)
    return lidar_points

def transform_points_to_voxels(points, voxel_generator):
    voxel_output = voxel_generator.generate(points)
    voxels, coordinates, num_points = voxel_output

    data_dict = {}
    data_dict['voxels'] = voxels
    data_dict['voxel_coords'] = coordinates
    data_dict['voxel_num_points'] = num_points
    return data_dict

def _subporc_voxelize(param):
    args = param.args
    sequence_dir = param.sequence_dir
    dataset_base_dir = param.dataset_base_dir
    voxel_output_dir = param.voxel_output_dir
    voxel_size = param.voxel_size
    # voxel_generator = param.voxel_generator
    config = param.config

    # init voxel generator
    voxel_generator = VoxelGeneratorWrapper(
        vsize_xyz=voxel_size,
        coors_range_xyz=config.single_chip_mode.lidar.pc_range,
        num_point_features=3,
        max_num_points_per_voxel=config.single_chip_mode.lidar.voxel_max_num_points,
        max_num_voxels=config.single_chip_mode.lidar.max_voxels
    )

    if args.mode == "sc":
        lidar_dir = dataset_base_dir / sequence_dir.name / "lidar_sc"
        voxel_link_dir = dataset_base_dir / sequence_dir.name / \
            f"voxel_sc_{round(voxel_size[0],2)}_{round(voxel_size[1],2)}_{round(voxel_size[2],2)}"
        voxel_dir = voxel_output_dir / sequence_dir.name / \
            f"voxel_sc_{round(voxel_size[0],2)}_{round(voxel_size[1],2)}_{round(voxel_size[2],2)}"
        
    elif args.mode == "cc":
        lidar_dir = dataset_base_dir / sequence_dir.name / "lidar_cc"
        voxel_link_dir = dataset_base_dir / sequence_dir.name / \
            f"voxel_cc_{round(voxel_size[0],2)}_{round(voxel_size[1],2)}_{round(voxel_size[2],2)}"
        voxel_dir = voxel_output_dir / sequence_dir.name / \
            f"voxel_cc_{round(voxel_size[0],2)}_{round(voxel_size[1],2)}_{round(voxel_size[2],2)}"
        
    elif args.mode == "sc_cone":
        lidar_dir = dataset_base_dir / sequence_dir.name / "lidar_sc"
        voxel_link_dir = dataset_base_dir / sequence_dir.name / \
            f"cone_sc_{round(voxel_size[0],2)}_{round(voxel_size[1],2)}_{round(voxel_size[2],2)}"
        voxel_dir = voxel_output_dir / sequence_dir.name / \
            f"cone_sc_{round(voxel_size[0],2)}_{round(voxel_size[1],2)}_{round(voxel_size[2],2)}"
    else:
        raise ValueError(f"Unknown mode {args.mode}")
    if not lidar_dir.exists():
        raise ValueError(f"lidar_dir {lidar_dir} not exists")
    ensure_path_exists(voxel_dir)

    lidar_files = sorted(lidar_dir.glob("*.bin"))
    print(f"Found {len(lidar_files)} lidar files in {lidar_dir}")

    for lidar_file in (lidar_files):
        voxel_file = voxel_dir / f"{lidar_file.stem}.npy"
        if voxel_file.exists():
            continue
        lidar_points = load_lidar(lidar_file)
        if lidar_points is None or len(lidar_points) == 0:
            print(f"Skipping empty lidar file {lidar_file}")
            continue
        if args.mode == "sc_cone":
            # convert cartesian to polar coordinates
            lidar_points = cartesian2polar(lidar_points)

        data_dict = transform_points_to_voxels(lidar_points, voxel_generator)
        # save as npy file
        np.save(voxel_file, data_dict)

    # symlink voxel_dir to voxel_link_dir
    if not voxel_link_dir.exists():
        print(f"Creating voxel link directory {voxel_link_dir}")
        os.symlink(voxel_dir, voxel_link_dir)
    else:
        print(f"voxel_link_dir {voxel_link_dir} already exists")
        raise ValueError(f"voxel_link_dir {voxel_link_dir} already exists")

    ########## DEBUG ##########
    # # delete the out dir
    # if voxel_link_dir.exists():
    #     os.remove(voxel_link_dir)
    # import shutil
    # if voxel_dir.exists():
    #     shutil.rmtree(voxel_dir)
    ########## DEBUG ##########

    return

def main():
    args = arg_parser()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = EasyDict(config)

    dataset_base_dir = Path(config.output_dir)
    voxel_output_dir = Path(config.voxel_output_dir)
    sequence_dirs = [d for d in dataset_base_dir.iterdir() if d.is_dir()]
    print(f"Found {len(sequence_dirs)} sequences in {dataset_base_dir}")

    voxel_size = np.array(config.single_chip_mode.lidar.voxel_size)

    params = []
    for sequence_dir in (sequence_dirs):
        # prepare the parameters for each sequence
        param = EasyDict()
        param.args = args
        param.sequence_dir = sequence_dir
        param.dataset_base_dir = dataset_base_dir
        param.voxel_output_dir = voxel_output_dir
        param.voxel_size = voxel_size
        # param.voxel_generator = voxel_generator
        param.config = config
        params.append(param)

    # process each sequence in parallel
    num_workers = config.num_workers
    print(f"Processing {len(params)} sequences in parallel with {num_workers} workers")
    imap_tqdm(
        _subporc_voxelize,
        params,
        processes=num_workers,
    )


if __name__ == "__main__":
    main()
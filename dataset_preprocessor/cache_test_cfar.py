'''
This script is used to preprocess the radar data in the Coloradar dataset with the following steps:
    1. Load the radar data
    2. Transform the lidar data from lidar coord to radar coord
    3. Select the overlapping time stamps of radar and lidar
    3. Crop the lidar data based on the field of view (FOV) of the radar
    4. Save the preprocessed radar data
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
from utils.utils import ensure_path_exists
from utils.concurrent import imap_tqdm
import json

from dataset_preprocessor.cache_test_cfar_utils import read_radar_map_bin, rae_interpo, \
                                                    RA2DDetectorTensor, cube_idx2coord
from dataset_preprocessor.lidar import filter_points_polar, save_lidar_data
import torch as th

def arg_parser():
    parser = argparse.ArgumentParser(description="Convert Radar data to Lidar data")
    parser.add_argument("--config", type=str, 
                        default="dataset_preprocessor/config/coloradar_config_test_set.yaml",)
    parser.add_argument("--mode", type=str, 
                        default="sc", choices=["sc", "cc"],
                        help="Mode to convert data, sc: single-chip radar, cs: cascade radar")
    return parser.parse_args()

def antenna_array(file_path):
    rxl = []    # RX layout
    txl = []    # TX layout
    with open(os.path.join(file_path), "r") as fh:
        for line in fh:
            if line.startswith("# "):
                continue
            else:
                chunks = line.strip().split(" ")
                if chunks[0] == "rx":
                    rxl.append([int(x) for x in chunks[1:]])
                elif chunks[0] == "tx":
                    txl.append([int(x) for x in chunks[1:]])
                else:
                    continue

    txl = np.array(txl)
    rxl = np.array(rxl)
    return txl, rxl

def save_radarcube(radarcube:np.array, save_path:Path)->None:
    # save as binary file
    if isinstance(radarcube, th.Tensor):
        radarcube = radarcube.detach().cpu().numpy()
    radarcube = radarcube.astype(np.float32)
    radarcube.tofile(save_path)
    return

def load_radar_data(radar_config, radar_path:Path):
    loaded_data = np.fromfile(radar_path,dtype = "int16")
    loaded_data_numpy_adc = loaded_data.reshape((radar_config.numTxChan, radar_config.numRxChan, 
                                                 radar_config.numChirpsPerFrame, radar_config.numAdcSamples, 2))
    
    I = loaded_data_numpy_adc[:, :, :, :, 0]
    Q = loaded_data_numpy_adc[:, :, :, :, 1]
    loaded_data_numpy_adc= I + 1j * Q
    # remove the DC component
    loaded_data_numpy_adc -= np.mean(loaded_data_numpy_adc)

    return loaded_data_numpy_adc

def _subproc_process_radar(params):
    """
    Process the radar data in a subprocess.
    Args:
        params (EasyDict): Parameters for processing the radar data.
    Returns:
        None
    """
    spectrum_files = params.spectrum_files
    out_dir = params.out_dir
    radar_config = params.radar_config

    for i, _ in tqdm(enumerate(spectrum_files)):
        # 1. load spectrum_files
        radar_map = read_radar_map_bin(spectrum_files[i], config=radar_config) 
        radar_map = radar_map[:, :, :, 0].unsqueeze(0)  # only take the intensity channel

        # 2. upsample the radar spectrum
        up_radar_map = rae_interpo(radar_map, radar_config.target_r_size, 
                                   radar_config.target_a_size, radar_config.target_e_size)

        # 3. conduct CFAR process
        all_peaks = RA2DDetectorTensor(up_radar_map, num=radar_config.cfar_num_point)

        # 4. get radar point from selected index of radar spectrum
        selected_coords = cube_idx2coord(all_peaks, radar_config, return_in_degrees=True)   # in polar coords
        selected_coords = filter_points_polar(selected_coords, radar_config.fov)

        # # 5. save the radar point in polar coords
        out_dir_i = out_dir / f"{i:04d}.bin"
        save_lidar_data(selected_coords, out_dir_i)

def main():
    args = arg_parser()
    with open(args.config, 'r') as config_file:
        config = yaml.load(config_file, yaml.FullLoader)
    config = EasyDict(config)
    print(config)

    dataset_dir = Path(config.root_dir)
    out_base_dir = Path(config.output_dir)
    split_file = config.get('split_file', None)
    if split_file == None:
        raise ValueError('You are using the wrong config file!')
    split_file_path = out_base_dir / split_file
    print('Using split file: ' + str(split_file_path))
    with open(split_file_path, 'r') as f:
            split_content = json.load(f)
    seq_list = split_content[config.split]

    sequence_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name in seq_list]
    print(f"Found {len(sequence_dirs)} sequences in {dataset_dir}")

    # read radar config
    radar_config_file_path = config.single_chip_mode.radar.config
    with open(radar_config_file_path, 'r', encoding="utf-8") as fid:
        radar_config = EasyDict(yaml.load(fid, Loader=yaml.FullLoader))  
    radar_config.chirpRampTime = radar_config.SamplePerChripUp / radar_config.Fs
    radar_config.chirpBandwidth = radar_config.Kr * radar_config.chirpRampTime
    radar_config.max_range = (3e8 * radar_config.chirpRampTime * radar_config.Fs) / (2 * radar_config.chirpBandwidth ) 
    radar_config.fov = [
        [0, radar_config.max_range],
        radar_config.angles_DOA_az, 
        radar_config.angles_DOA_ele
    ]

    radar_config.target_r_size = config.single_chip_mode.radar.cfar.tgt_r_dim
    radar_config.target_a_size = config.single_chip_mode.radar.cfar.tgt_a_dim
    radar_config.target_e_size = config.single_chip_mode.radar.cfar.tgt_e_dim

    radar_config.input_r_size = config.single_chip_mode.radar.cfar.input_r_dim
    radar_config.input_a_size = config.single_chip_mode.radar.cfar.input_a_dim
    radar_config.input_e_size = config.single_chip_mode.radar.cfar.input_e_dim

    radar_config.cfar_num_point = int(float(config.single_chip_mode.radar.cfar.cfar_num_point))

    params_list = []
    for seq_dir in (sequence_dirs):
        if args.mode == "sc":
            cube_dir = out_base_dir / seq_dir.name / "single_chip" / "radarcube_high_res"
            spectrum_files = list(cube_dir.glob("*.bin"))
            spectrum_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
            out_dir = out_base_dir / seq_dir.name / "single_chip" / "radar_cfar_low_thrd"
            ensure_path_exists(out_dir)
            print(f"Found {len(spectrum_files)} radar files in {seq_dir.name}")

            # init params
            params = EasyDict()
            params.spectrum_files = spectrum_files
            params.out_dir = out_dir
            params.radar_config = radar_config
            params_list.append(params)

        elif args.mode == "cc":
            NotImplemented  
        
    imap_tqdm(
        _subproc_process_radar,
        params_list,
        processes=config.num_workers,
        desc="Processing radar data",
    )

if __name__ == "__main__":
    main()
    
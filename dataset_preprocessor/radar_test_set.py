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
from constants import T_RADAR_TO_LIDAR, NUMBER_RECORDING_ATTRIBUTES, EXCLUDE_DIR_NAMES
from utils.utils import ensure_path_exists
from utils.concurrent import imap_tqdm
import json

from dataset_preprocessor.utils.radar_preprocessing import RAEIVVmap
import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
    # print("loaded_data", loaded_data.shape)
    # print(radar_config.numTxChan, radar_config.numRxChan,radar_config.numChirpsPerFrame, radar_config.numAdcSamples)
    loaded_data_numpy_adc = loaded_data.reshape((radar_config.numTxChan, radar_config.numRxChan, 
                                                 radar_config.numChirpsPerFrame, radar_config.numAdcSamples, 2))
    # print(loaded_data_numpy_adc.shape)
    # print("loaded_data_numpy_adc", type(loaded_data_numpy_adc))
    
    I = loaded_data_numpy_adc[:, :, :, :, 0]
    Q = loaded_data_numpy_adc[:, :, :, :, 1]
    loaded_data_numpy_adc= I + 1j * Q
    # 去除直流分量
    loaded_data_numpy_adc -= np.mean(loaded_data_numpy_adc)

    return loaded_data_numpy_adc

def load_npy_radar_data(radar_config, radar_path:Path):
    loaded_data = np.load(radar_path, allow_pickle=True)
    # print("loaded_data", loaded_data.shape)
    # print(radar_config.numTxChan, radar_config.numRxChan,radar_config.numChirpsPerFrame, radar_config.numAdcSamples)
    loaded_data_numpy_adc = loaded_data.reshape((radar_config.numTxChan, radar_config.numRxChan, 
                                                 radar_config.numChirpsPerFrame, radar_config.numAdcSamples))
    # 去除直流分量
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
    adc_files = params.adc_files
    out_dir = params.out_dir
    seq_dir = params.seq_dir
    radar_config = params.radar_config
    tx_array = params.tx_array
    rx_array = params.rx_array

    # get radar index
    index_file = Path(seq_dir / "single_chip" / "adc_samples" / "radar_index_sequence.txt")
    with open(index_file, 'r') as f:
        rindex = f.readlines()
    rindex = [int(i) for i in rindex]
    print(f"Found {len(rindex)} overlapping time stamps of lidar and radar")

    # ####### debug #######
    # import shutil
    # print(f"Delete the out dir {out_dir}")
    # if out_dir.exists():
    #     shutil.rmtree(out_dir)
    #     print(f"Delete the out dir {out_dir}")
    # ######## debug #######

    for i, index in tqdm(enumerate(rindex)):
        adc_file = adc_files[index]
        # 1. load spectrum_files
        if adc_file.suffix == ".bin":
            adc_data = load_radar_data(radar_config, adc_file)  # (Tx, Rx, Chirp, Sample)
        elif adc_file.suffix == ".npy":
            adc_data = load_npy_radar_data(radar_config,adc_file)     # (Tx, Rx, Chirp, Sample)
        RAEmap = RAEIVVmap(adc_data, radar_config, tx_array, rx_array)  #(range, azimuth, elevation, 3)               

        # save radarcube
        out_dir_i = out_dir / f"{i:04d}.bin"
        save_radarcube(RAEmap, out_dir_i)


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

    # sequence_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name not in EXCLUDE_DIR_NAMES]
    sequence_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and d.name in seq_list]
    # sequence_dirs = [Path("/data/ruijiezzz/dataset/coloradar/12_21_2020_ec_hallways_run0")]   #debug
    print(f"Found {len(sequence_dirs)} sequences in {dataset_dir}")

    params_list = []
    for seq_dir in (sequence_dirs):
        if args.mode == "sc":
            adc_dir = seq_dir / "single_chip" / "adc_samples" / "data"
            adc_files = list(adc_dir.glob("*.bin"))
            # adc_files = list(adc_dir.glob("*.npy"))
            adc_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
            out_dir = out_base_dir / seq_dir.name / "single_chip" / "radarcube_high_res"
            ensure_path_exists(out_dir)
            print(f"Found {len(adc_files)} radar files in {seq_dir.name}")

            # read radar config
            radar_config_file_path = config.single_chip_mode.radar.config
            with open(radar_config_file_path, 'r', encoding="utf-8") as fid:
                radar_config = EasyDict(yaml.load(fid, Loader=yaml.FullLoader))  
            radar_config.chirpRampTime = radar_config.SamplePerChripUp / radar_config.Fs
            radar_config.chirpBandwidth = radar_config.Kr * radar_config.chirpRampTime
            radar_config.max_range = (3e8 * radar_config.chirpRampTime * radar_config.Fs) / (2 * radar_config.chirpBandwidth )  

            # get antenna array
            antenna_file_path = config.single_chip_mode.radar.antenna_file_path
            tx_array, rx_array = antenna_array(antenna_file_path)


            # init params
            params = EasyDict()
            params.adc_files = adc_files
            params.out_dir = out_dir
            params.seq_dir = seq_dir
            params.radar_config = radar_config
            params.tx_array = tx_array
            params.rx_array = rx_array
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
    
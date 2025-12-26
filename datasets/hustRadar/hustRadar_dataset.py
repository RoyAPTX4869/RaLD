from datasets.aligned_coloradar.Coloradar_dataset import ColoRadarDataset
from easydict import EasyDict
from pathlib import Path
import numpy as np
import json
from datasets.utils.voxelize import VoxelGeneratorWrapper

class HUSTRadarDataset(ColoRadarDataset):
    def __init__(self, root_dir:Path, config:EasyDict, radar_type='scRadar', loader_type='train'):
        super().__init__(root_dir, config, radar_type, loader_type)
        self.root_dir = Path(root_dir)
        self.config = config
        self.radar_type = radar_type
        assert loader_type in ['train', 'val', 'test'], f"Invalid loader type {loader_type}"
        assert radar_type in ['scRadar', 'ccRadar'], f"Invalid radar type {radar_type}"
        self.split_file = config.split_file
        self.loader_type = loader_type
        self.shuffle_pts = config.lidar.get('shuffle_pts', False)

        self.load_query = True
        self.load_radar = True
        self.use_cache_latent = config.get('use_cache_latent', False)
        if self.use_cache_latent:
            self.cache_latent_dir = Path(config.cache_latent_base_dir) / Path(config.cache_latent_sub_dir)
            assert self.cache_latent_dir.exists(), f"Cache latent dir {self.cache_latent_dir} does not exist"

        self.use_pred_latent = config.get('use_pred_latent', False) and self.loader_type == 'test'
        if self.use_pred_latent:
            self.pred_latent_dir = Path(config.pred_latent_base_dir) / Path(config.pred_latent_sub_dir)
            assert self.pred_latent_dir.exists(), f"Pred latent dir {self.pred_latent_dir} does not exist"

        self.use_query_helper = config.get("use_query_helper", False) and self.loader_type == 'test'
        if self.use_query_helper:
            self.query_helper_aug = config.get("query_helper_aug", False)
            self.query_aug_num = int(float(config.get("query_aug_num", 0)))
            self.query_aug_scale = int(config.get('query_aug_scale', 2))

        self.split = self.load_split()
        # lidar config
        self.norm_isotropy = self.config.lidar.norm_isotropy
        self.norm_anisotropy = self.config.lidar.norm_anisotropy
        self.query_ratio = self.config.lidar.query_ratio
        self.lidar_pc_range = np.array(self.config.lidar.pc_range)
        self.lidar_feat_channels = self.config.lidar.num_point_features
        self.sampling = self.config.lidar.sampling
        self.num_samples = self.config.lidar.num_samples
        grid_size = (self.lidar_pc_range[3:6] - self.lidar_pc_range[0:3]) / np.array(self.config.lidar.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.voxel_size = self.config.lidar.voxel_size
        self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.lidar_pc_range,
                num_point_features=self.lidar_feat_channels,
                max_num_points_per_voxel=self.config.lidar.max_points_per_voxel,
                max_num_voxels=self.config.lidar.max_number_of_voxels,
            )
        print(f"Loaded {loader_type} HustRadar dataset successfully")

    def load_split(self):
        split_file = self.root_dir / self.split_file
        print('Using split file: ' + str(self.split_file))
        with open(split_file, 'r') as f:
            split = json.load(f)
        self.split = split
        self.seq_list = self.split[self.loader_type]

        # check if all sequences exist
        for seq in self.seq_list:
            lidar_dir_name = "lidar_sc" if self.radar_type == 'scRadar' else "lidar_cc"
            assert (self.root_dir / seq / lidar_dir_name).exists(), f"Path {self.root_dir / seq / lidar_dir_name} does not exist"
            # radar_dir_name = "single_chip/radarcube" if self.radar_type == 'scRadar' else "cascade/radarcube"
            radar_dir_name = "rae_map"
            assert (self.root_dir / seq / radar_dir_name).exists(), f"Path {self.root_dir / seq / radar_dir_name} does not exist"

        # serialize the index_dict and lidar_path_list
        total_num_samples = 0
        self.index_dict = {}
        for seq in self.seq_list:
            seq_sample= len(list((self.root_dir / seq / lidar_dir_name).glob('*.bin')))
            self.index_dict[seq] = (total_num_samples, total_num_samples + seq_sample)
            total_num_samples += seq_sample
        self.lidar_path_list = []
        for seq in self.seq_list:
            seq_lidar_path_list = list((self.root_dir / seq / lidar_dir_name).glob('*.bin'))
            seq_lidar_path_list.sort()
            self.lidar_path_list.extend(seq_lidar_path_list)

        self.cache_voxel = self.config.lidar.cache_voxel
        if self.cache_voxel:
            self.lidar_voxel_path_list = []
            voxel_size = self.config.lidar.voxel_size
            type_name = 'sc' if self.radar_type == 'scRadar' else 'cc'
            voxel_type = 'cone' if self.config.lidar.get('view_cone_mode', False) else 'voxel'
            target_voxel_dir = f"{voxel_type}_{type_name}_{round(voxel_size[0],2)}_{round(voxel_size[1],2)}_{round(voxel_size[2],2)}"
            for seq in self.seq_list:
                assert (self.root_dir / seq / target_voxel_dir).exists(), \
                    f"Path {self.root_dir / seq / target_voxel_dir} does not exist"
                seq_voxel_path_list = list((self.root_dir / seq / target_voxel_dir).glob('*.npy'))
                seq_voxel_path_list.sort()
                self.lidar_voxel_path_list.extend(seq_voxel_path_list)
            assert len(self.lidar_path_list) == len(self.lidar_voxel_path_list), \
                f"Length of lidar_path_list {len(self.lidar_path_list)} and lidar_voxel_path_list {len(self.lidar_voxel_path_list)} do not match"
            print(f"Found {len(self.lidar_voxel_path_list)} voxel files in {self.root_dir}")
            
        # radar data serialization
        if self.load_radar:
            self.radar_path_list = []
            for seq in self.seq_list:
                radar_dir_name = "rae_map" 
                seq_radar_path_list = list((self.root_dir / seq / radar_dir_name).glob('*.npy'))
                seq_radar_path_list.sort(key=lambda x: int(x.stem))
                self.radar_path_list.extend(seq_radar_path_list)
            assert len(self.lidar_path_list) == len(self.radar_path_list), \
                f"Length of lidar_path_list {len(self.lidar_path_list)} and radar_path_list {len(self.radar_path_list)} do not match"
            
        # load cache latent if use_cache_latent
        if self.use_cache_latent:
            self.cache_latent_path_list = []
            for seq in self.seq_list:
                seq_cache_latent_path_list = list((self.cache_latent_dir / seq).glob('*.npz'))
                seq_cache_latent_path_list.sort()
                self.cache_latent_path_list.extend(seq_cache_latent_path_list)
            assert len(self.lidar_path_list) == len(self.cache_latent_path_list), \
                f"Length of lidar_path_list {len(self.lidar_path_list)} and cache_latent_path_list {len(self.cache_latent_path_list)} do not match"
            
        if self.use_pred_latent:
            self.pred_latent_path_list = []
            for seq in self.seq_list:
                seq_pred_latent_path_list = list((self.pred_latent_dir / seq / 'latent_tokens').glob('*.pt'))
                seq_pred_latent_path_list.sort()
                self.pred_latent_path_list.extend(seq_pred_latent_path_list)
            assert len(self.lidar_path_list) == len(self.pred_latent_path_list), \
                f"Length of lidar_path_list {len(self.lidar_path_list)} and pred_latent_path_list {len(self.pred_latent_path_list)} do not match"
            
        if self.use_query_helper:
            self.cfar_query_helper_path_list = []
            for seq in self.seq_list:
                helper_radar_dir_name = "single_chip/radar_cfar_low_thrd/" if self.radar_type == 'scRadar' else "cascade/radar_cfar_low_thrd/"
                seq_cfar_query_helper_path_list = list((self.root_dir / seq / helper_radar_dir_name).glob('*.bin'))
                seq_cfar_query_helper_path_list.sort()
                self.cfar_query_helper_path_list.extend(seq_cfar_query_helper_path_list)
            assert len(self.lidar_path_list) == len(self.cfar_query_helper_path_list), \
                f"Length of lidar_path_list {len(self.lidar_path_list)} and cfar_query_helper_path_list {len(self.cfar_query_helper_path_list)} do not match"

        return 
    
    def load_radarcube(self, idx):
        radar_path = self.radar_path_list[idx]
        try:
            radar_cube = np.load(radar_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading radar file {radar_path}")
            print(e)
            raise ValueError(f"Error loading radar file {radar_path}")
        radar_cube = radar_cube.reshape(self.config.radar.input_r_dim, self.config.radar.input_a_dim, \
                                        self.config.radar.input_e_dim, -1)
        return radar_cube, radar_path
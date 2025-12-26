"""
This module contains the implementation of the aligned coloRadar dataset class.
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from easydict import EasyDict
from pathlib import Path
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from datasets.utils.voxelize import VoxelGeneratorWrapper
from datasets.utils.query_helper import aug_query_helper
from dataset_preprocessor.lidar import cartesian2polar

class ColoRadarDataset(Dataset):
    def __init__(self, root_dir:Path, config:EasyDict, radar_type='scRadar', loader_type='train'):
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
        print(f"Loaded {loader_type} ColoRadar dataset successfully")
        return
        
    def __getitem__(self, index):
        '''
        Returns:
            data_dict: dict, which contains the following keys:
                lidar_points: torch.Tensor, (N, 3), lidar points
                query_points: torch.Tensor, (N, 3), query points
                query_labels: torch.Tensor, (N,), query labels
                in_voxel_num: int, number of points in the voxel
                grid_points: torch.Tensor, (N, 3), grid points
        '''
        data_dict = {}
        points, lidar_path = self.load_lidar(index)
        if self.loader_type != 'train':
            data_dict['raw_lidar_points'] = points
        if self.shuffle_pts:
            points = self.points_shuffle(points)
        data_dict['lidar_path'] = [str(lidar_path)]
        if self.config.lidar.get('view_cone_mode', False):
            points = cartesian2polar(points)
        voxels, voxel_coords, voxel_num_points = self.transform_points_to_voxels(points, self.config.lidar, idx=index)

        if self.sampling:
            try:
                points_ind = np.random.default_rng().choice(points.shape[0], self.num_samples, replace=False)
                points = points[points_ind]
            except:
                print(f"Error sampling points from {points.shape[0]} points")
                raise ValueError(f"Error sampling points from {points.shape[0]} points")
            
        self.points_num = points.shape[0]
        self.in_num = int(self.points_num*self.query_ratio)
        self.out_num = self.points_num - self.in_num
        data_dict['lidar_points'] = torch.from_numpy(points[:,:3])
        if self.load_query:
            data_dict = self.transform_voxels_to_query_points(voxels,voxel_coords,data_dict)
            data_dict['in_voxel_num'] = self.in_num

        if self.use_query_helper:
            helper_point, helper_point_path = self.load_helper_point(index)
            data_dict['helper_points'] = helper_point
            data_dict['helper_point_path'] = [str(helper_point_path)]

        data_dict = self.norm_points(data_dict)

        if self.load_radar:
            radar_data, radar_path = self.load_radarcube(index)
            data_dict['radar_cube'] = self.process_radar_data(radar_data)
            data_dict['radar_path'] = [str(radar_path)]

        if self.use_cache_latent:
            cached_latent, cache_latent_path = self.load_cached_latent(index)
            data_dict['cache_latent'] = cached_latent
            data_dict['cache_latent_path'] = [str(cache_latent_path)]

            assert lidar_path.name == cache_latent_path.stem, \
                f"Cache latent path {cache_latent_path} does not match lidar path {lidar_path}"

        if self.use_pred_latent:
            pred_latent, pred_latent_path = self.load_pred_latent(index)
            data_dict['pred_latent'] = pred_latent
            data_dict['pred_latent_path'] = [str(pred_latent_path)]

            assert lidar_path.stem == pred_latent_path.stem, \
                f"Pred latent path {pred_latent_path} does not match lidar path {lidar_path}"

        return data_dict
    
    def __len__(self):
        return len(self.lidar_path_list)

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
            radar_dir_name = "single_chip/radarcube_raw" if self.radar_type == 'scRadar' else "cascade/radarcube_raw"
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
                radar_dir_name = "single_chip/radarcube_raw" if self.radar_type == 'scRadar' else "cascade/radarcube_raw"
                seq_radar_path_list = list((self.root_dir / seq / radar_dir_name).glob('*.bin'))
                seq_radar_path_list.sort()  
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
    
    def load_lidar(self, idx):
        lidar_path = self.lidar_path_list[idx]
        try:
            lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, self.lidar_feat_channels)
        except Exception as e:
            print(f"Error loading lidar file {lidar_path}")
            print(e)
        return lidar_points, lidar_path

    def transform_voxels_to_query_points(self,voxel_features,coords,data_dict):
        self.voxel_x = self.voxel_size[0]
        self.voxel_y = self.voxel_size[1]
        self.voxel_z = self.voxel_size[2]
        self.x_offset = self.voxel_x / 2 + self.lidar_pc_range[0]
        self.y_offset = self.voxel_y / 2 + self.lidar_pc_range[1]
        self.z_offset = self.voxel_z / 2 + self.lidar_pc_range[2]

        coords = coords[:, [2, 1, 0]] #(0:X, 1:Y, 2:Z)
        
        # Get the center of the voxel
        f_center = torch.zeros_like(voxel_features[:, 0, :3])
        f_center[:, 0] =  coords[:, 0].to(f_center.dtype) * self.voxel_x + self.x_offset
        f_center[:, 1] =  coords[:, 1].to(f_center.dtype) * self.voxel_y + self.y_offset
        f_center[:, 2] =  coords[:, 2].to(f_center.dtype) * self.voxel_z + self.z_offset

        # Convert voxel_size list to numpy array for arithmetic operations
        voxel_size = np.array(self.voxel_size)
        # random sample points in the voxel
        if self.loader_type == 'train':
            coords_in_voxel = np.random.default_rng().uniform(low=-voxel_size/2, high=voxel_size/2, size=(self.in_num, 3))
            coords_out_voxel = np.random.default_rng().uniform(low=-voxel_size/2, high=voxel_size/2, size=(self.out_num, 3))
            random_voxel_index = np.random.default_rng().choice(voxel_features.shape[0], self.in_num, replace=True)

            # change to torch
            coords_in_voxel = torch.from_numpy(coords_in_voxel).type_as(voxel_features)
            coords_out_voxel = torch.from_numpy(coords_out_voxel).type_as(voxel_features)
            random_voxel_index = torch.from_numpy(random_voxel_index).type_as(voxel_features).long()

            # Get the coordinates of the occupied voxel and the empty voxel
            occupied_voxel_coords = f_center[random_voxel_index]
            points_in_voxel = occupied_voxel_coords + coords_in_voxel
            empty_voxel_coords = self.get_empty_voxel_centers(coords, self.out_num)
            points_out_voxel = empty_voxel_coords + coords_out_voxel

            # label in voxel:1, out voxel:0
            label_in_voxel = torch.ones(self.in_num)
            label_out_voxel = torch.zeros(self.out_num)

            # concat in and out
            query_points = torch.cat([points_in_voxel, points_out_voxel], dim=0)
            query_labels = torch.cat([label_in_voxel, label_out_voxel], dim=0)
        else:
            coords_in_voxel = np.random.default_rng().uniform(low=-voxel_size/2, high=voxel_size/2, size=(self.points_num, 3))
            random_voxel_index = np.random.default_rng().choice(voxel_features.shape[0], self.points_num, replace=True)

             # change to torch
            coords_in_voxel = torch.from_numpy(coords_in_voxel).type_as(voxel_features)
            random_voxel_index = torch.from_numpy(random_voxel_index).type_as(voxel_features).long()

            # Get the coordinates of the occupied voxel 
            occupied_voxel_coords = f_center[random_voxel_index]
            query_points = occupied_voxel_coords + coords_in_voxel
            query_labels = torch.ones(self.points_num)

        data_dict['query_points'] = query_points
        data_dict['query_labels'] = query_labels
        return data_dict

    def transform_points_to_voxels(self, points, config=None, idx=None):
        if self.cache_voxel:
            voxel_path = self.lidar_voxel_path_list[idx]
            if not os.path.exists(voxel_path):
                print(f"Voxel path {voxel_path} does not exist")
                raise FileNotFoundError
            voxel_dict = np.load(voxel_path, allow_pickle=True).item()
            voxels = voxel_dict['voxels']
            coordinates = voxel_dict['voxel_coords']
            num_points = voxel_dict['voxel_num_points']
        else:
            voxel_output = self.voxel_generator.generate(points)
            voxels, coordinates, num_points = voxel_output

        data_dict = {}
        if config.get('DOUBLE_FLIP', False):
            voxels_list, voxel_coords_list, voxel_num_points_list = [voxels], [coordinates], [num_points]
            points_yflip, points_xflip, points_xyflip = self.double_flip(points)
            points_list = [points_yflip, points_xflip, points_xyflip]
            keys = ['yflip', 'xflip', 'xyflip']
            for i, key in enumerate(keys):
                voxel_output = self.voxel_generator.generate(points_list[i])
                voxels, coordinates, num_points = voxel_output

                if not data_dict['use_lead_xyz']:
                    voxels = voxels[..., 3:]
                voxels_list.append(voxels)
                voxel_coords_list.append(coordinates)
                voxel_num_points_list.append(num_points)

            voxels = torch.from_numpy(voxels_list)
            voxel_coords = torch.from_numpy(voxel_coords_list)
            voxel_num_points = torch.from_numpy(voxel_num_points_list)
        else:
            voxels = torch.from_numpy(voxels)
            voxel_coords = torch.from_numpy(coordinates)
            voxel_num_points = torch.from_numpy(num_points)
        return voxels,voxel_coords,voxel_num_points

    def get_empty_voxel_centers(self, coords, empty_voxel_num):
        device = coords.device
        
        # change to tuple
        grid_size = tuple(self.grid_size)
        
        # generate grid coordinates
        x = torch.arange(grid_size[0], device=device)
        y = torch.arange(grid_size[1], device=device)
        z = torch.arange(grid_size[2], device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')

        occupied = torch.zeros(grid_size, dtype=torch.bool, device=device)
        coords_idx = coords.long()
        occupied[coords_idx[:, 0], coords_idx[:, 1], coords_idx[:, 2]] = True
        empty_mask = ~occupied.flatten()
        
        empty_indices = torch.nonzero(empty_mask).squeeze()
        perm = torch.randint(0, len(empty_indices), (empty_voxel_num,), device=device)
        selected_empty_coords_x = grid_x.flatten()[empty_indices[perm]]
        selected_empty_coords_y = grid_y.flatten()[empty_indices[perm]]
        selected_empty_coords_z = grid_z.flatten()[empty_indices[perm]]
        
        empty_voxel_centers = torch.zeros((empty_voxel_num, 3), dtype=torch.float32, device=device)
        empty_voxel_centers[:, 0] = selected_empty_coords_x * self.voxel_x + self.x_offset
        empty_voxel_centers[:, 1] = selected_empty_coords_y * self.voxel_y + self.y_offset
        empty_voxel_centers[:, 2] = selected_empty_coords_z * self.voxel_z + self.z_offset

        return empty_voxel_centers

    def norm_points(self, data_dict):
        points = data_dict['lidar_points']
        query_points = data_dict.get('query_points', None)
        grid_points = data_dict.get('grid_points', None)
        helper_points = data_dict.get('helper_points', None)
        x_offset = (self.lidar_pc_range[3] + self.lidar_pc_range[0]) / 2
        y_offset = (self.lidar_pc_range[4] + self.lidar_pc_range[1]) / 2
        z_offset = (self.lidar_pc_range[5] + self.lidar_pc_range[2]) / 2
        x_scale = (self.lidar_pc_range[3] - self.lidar_pc_range[0]) / 2
        y_scale = (self.lidar_pc_range[4] - self.lidar_pc_range[1]) / 2
        z_scale = (self.lidar_pc_range[5] - self.lidar_pc_range[2]) / 2
        if self.norm_anisotropy:  # anisotropy scaling, normalize x, y, z to [-1, 1]
            points[:, 0] = (points[:, 0] - x_offset) / x_scale
            points[:, 1] = (points[:, 1] - y_offset) / y_scale
            points[:, 2] = (points[:, 2] - z_offset) / z_scale
            if query_points is not None:
                data_dict['raw_query_points'] = query_points
                query_points[:, 0] = (query_points[:, 0] - x_offset) / x_scale
                query_points[:, 1] = (query_points[:, 1] - y_offset) / y_scale
                query_points[:, 2] = (query_points[:, 2] - z_offset) / z_scale
                data_dict['query_points'] = query_points
            if grid_points is not None:
                grid_points = data_dict['grid_points']
                grid_points[:, 0] = (grid_points[:, 0] - x_offset) / x_scale
                grid_points[:, 1] = (grid_points[:, 1] - y_offset) / y_scale
                grid_points[:, 2] = (grid_points[:, 2] - z_offset) / z_scale
                data_dict['grid_points'] = grid_points
            if helper_points is not None:
                helper_points = data_dict['helper_points']
                helper_points[:, 0] = (helper_points[:, 0] - x_offset) / x_scale
                helper_points[:, 1] = (helper_points[:, 1] - y_offset) / y_scale
                helper_points[:, 2] = (helper_points[:, 2] - z_offset) / z_scale
                data_dict['helper_points'] = helper_points
            data_dict['lidar_points'] = points
        
        # isotropy scaling, normalize x, y, z to [-1, 1] with the same scale, which is the max scale
        if self.norm_isotropy:  
            max_scale = max(x_scale, y_scale, z_scale)
            offset = np.array([x_offset, y_offset, z_offset])
            points[:, :3] = (points[:, :3] - offset) / max_scale
            if query_points is not None:
                data_dict['raw_query_points'] = query_points
                query_points[:, :3] = (query_points[:, :3] - offset) / max_scale   
                data_dict['query_points'] = query_points
            if grid_points is not None:
                grid_points = data_dict['grid_points']
                grid_points = (grid_points - offset) / max_scale
                data_dict['grid_points'] = grid_points
            if helper_points is not None:
                helper_points = data_dict['helper_points']
                helper_points = (helper_points - offset) / max_scale
                data_dict['helper_points'] = helper_points
            data_dict['lidar_points'] = points
        return data_dict
    
    def load_radarcube(self, idx):
        radar_path = self.radar_path_list[idx]
        try:
            radar_cube = np.fromfile(radar_path, dtype=np.float32)
        except Exception as e:
            print(f"Error loading radar file {radar_path}")
            print(e)
            raise ValueError(f"Error loading radar file {radar_path}")
        radar_cube = radar_cube.reshape(self.config.radar.input_r_dim, self.config.radar.input_a_dim, \
                                        self.config.radar.input_e_dim, -1)
        return radar_cube, radar_path
    
    def process_radar_data(self, radar_cube, early_return=False):
        '''
        Process the radar data to conduct the following operations:
            1. Normalize the intensity & doppler values
            2. Mask the doppler values according to the last channel
            3. Upsample the radar cube
        Args:
            radar_cube: np.ndarray, (R, A, E, 3), radar cube
        Returns:
            radar_cube: np.ndarray, (R, A_up, E_up, 2), processed radar cube
        '''
        out_radar = np.zeros((self.config.radar.input_r_dim, self.config.radar.input_a_dim, self.config.radar.input_e_dim, 2)\
                             , dtype=np.float32)

        # Normalize the intensity & doppler values
        if self.config.radar.norm_intensity:
            max_intensity = self.config.radar.max_intensity
            # trunct normalize the intensity with the max intensity
            radar_cube[:,:,:,0] = np.clip(radar_cube[:,:,:,0], 0, max_intensity)
            out_radar[:,:,:,0] = radar_cube[:,:,:,0] / max_intensity

        # Mask the doppler values according to the last channel
        mask = radar_cube[:,:,:,-1]  # 1 for valid, 0 for invalid
        out_radar[:,:,:,1] = radar_cube[:,:,:,1] * mask

        # early return to skip upsample process
        if early_return:
            return out_radar
        
        if self.config.radar.norm_dopp:
            out_radar[:,:,:,1] = out_radar[:,:,:,1] / self.config.radar.max_dopp

        # Upsample the radar cube
        if self.config.radar.get('upsample', False):
            assert self.config.radar.input_r_dim == self.config.radar.tgt_r_dim, \
                f"Input radar cube r_dim {self.config.radar.input_r_dim} and target r_dim {self.config.radar.tgt_r_dim} do not match"
            out_radar_tensor_i = torch.from_numpy(out_radar[:,:,:,0]).unsqueeze(0)  # [1, R, A, E]
            out_radar_tensor_i = F.interpolate(out_radar_tensor_i, size=(self.config.radar.tgt_a_dim, self.config.radar.tgt_e_dim), \
                                                mode='bilinear', align_corners=True).squeeze(0).numpy() #
            out_radar_tensor_d = torch.from_numpy(out_radar[:,:,:,1]).unsqueeze(0)  # [1, R, A, E]
            out_radar_tensor_d = F.interpolate(out_radar_tensor_d, size=(self.config.radar.tgt_a_dim, self.config.radar.tgt_e_dim), \
                                            mode='bilinear', align_corners=True).squeeze(0).numpy()
            out_radar = np.stack((out_radar_tensor_i, out_radar_tensor_d), axis=-1)  # [R, A, E, 2]
        return out_radar

    def load_cached_latent(self, idx):
        cache_latent_path = self.cache_latent_path_list[idx]
        if not os.path.exists(cache_latent_path):
            print(f"Cache latent path {cache_latent_path} does not exist")
            raise FileNotFoundError
        cache_latent = np.load(cache_latent_path, allow_pickle=True)
        return torch.from_numpy(cache_latent['res_tokens']), cache_latent_path
    
    def load_pred_latent(self, index):
        pred_latent_path = self.pred_latent_path_list[index]
        if not os.path.exists(pred_latent_path):
            print(f"Pred latent path {pred_latent_path} does not exist")
            raise FileNotFoundError
        pred_latent = torch.load(pred_latent_path, weights_only=True)
        return pred_latent, pred_latent_path

    def load_helper_point(self, idx):
        helper_path = self.cfar_query_helper_path_list[idx]
        try:
            helper_points = np.fromfile(helper_path, dtype=np.float32).reshape(-1, self.lidar_feat_channels)
        except Exception as e:
            print(f"Error loading lidar file {helper_path}")
            print(e)
        if self.query_helper_aug:
            helper_points = aug_query_helper(helper_points, self.query_aug_num, self.lidar_pc_range, self.voxel_size)

        return torch.from_numpy(helper_points), helper_path

    def points_shuffle(self, points):
        '''
        Shuffle the points in the points array.
        Args:
            points: np.ndarray, (N, 3), points array
        Returns:
            points: np.ndarray, (N, 3), shuffled points array
        '''
        perm = np.random.permutation(points.shape[0])
        points = points[perm]
        return points

    def set_load_query(self, load_query:bool):
        self.load_query = load_query

    def set_load_radar(self, load_radar:bool):
        self.load_radar = load_radar

    def set_load_latent(self, use_cache_latent:bool):
        self.use_cache_latent = use_cache_latent

if __name__ == "__main__":
    import yaml
    import tqdm
    root_dir = Path('/data/ruijiezzz/dataset/processed_coloradar')
    config_dir = Path('configs/ar_vecset/ar_indoor_cfg_aniso_single_layer_mix_view_cone_unfreeze_enc_ints_only_eval_2.yml')
    with open(config_dir, 'r') as config_file:
        config = yaml.load(config_file, yaml.FullLoader)
    config = EasyDict(config)
    dataset = ColoRadarDataset(root_dir, config.dataset, radar_type='scRadar', loader_type='test')
    dataset.set_load_radar(True)
    dataset.set_load_latent(False)
    dataset.set_load_query(True)
    for i,data_dict in enumerate(tqdm.tqdm(dataset)):
        lidar_points = data_dict['lidar_points']
        if 'cache_latent' in data_dict:
            cache_latent = data_dict['cache_latent']
            print(i, "/", len(dataset), cache_latent.shape)
        # break
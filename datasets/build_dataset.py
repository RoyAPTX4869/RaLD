from datasets.aligned_coloradar import ColoRadarDataset as AlignedColoRadar
from datasets.hustRadar import HUSTRadarDataset as HUSTRadarDataset

def get_dataset(config, mode, sampler=None):
    if config.dataset_name == 'AlignedColoRadar':
        if mode == 'train':
            dataset = AlignedColoRadar(
                root_dir=config.root_dir,
                config=config,
                radar_type=config.radar_type,
                loader_type='train'
                )
        elif mode == 'val':
            dataset = AlignedColoRadar(
                root_dir=config.root_dir,
                config=config,
                radar_type=config.radar_type,
                loader_type='val'
                )
        elif mode == 'test':
            dataset = AlignedColoRadar(
                root_dir=config.root_dir,
                config=config,
                radar_type=config.radar_type,
                loader_type='test'
                )
    elif config.dataset_name == "HUSTRadarDataset":
        if mode == 'train':
            dataset = HUSTRadarDataset(
                root_dir=config.root_dir,
                config=config,
                radar_type='scRadar',
                loader_type='train'
                )
        elif mode == 'val':
            dataset = HUSTRadarDataset(
                root_dir=config.root_dir,
                config=config,
                radar_type='scRadar',
                loader_type='val'
                )
        elif mode == 'test':
            dataset = HUSTRadarDataset(
                root_dir=config.root_dir,
                config=config,
                radar_type='scRadar',
                loader_type='test'
                )       
    else:
        raise ValueError(f"Invalid dataset {config.dataset_name}")
    return dataset
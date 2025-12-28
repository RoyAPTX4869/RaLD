# RaLD | AAAI 2026 Oral
> Official implementation of [RaLD: Generating High-Resolution 3D Radar Point Clouds with Latent Diffusion](https://arxiv.org/pdf/2511.07067)

## Abstract
Millimeter-wave radar offers a promising sensing modality for autonomous systems thanks to its robustness in adverse conditions and low cost. However, its utility is significantly limited by the sparsity and low resolution of radar point clouds, which poses challenges for tasks requiring dense and accurate 3D perception. Despite that recent efforts have shown great potential by exploring generative approaches tocaddress this issue, they often rely on dense voxel representations that are inefficient and struggle to preserve structural detail. To fill this gap, we make the key observation that latent diffusion models (LDMs), though successful in other modalities, have not been effectively leveraged for radar-based 3D generation due to a lack of compatible representations and conditioning strategies. We introduce RaLD, a framework that bridges this gap by integrating scene-level frustum-based LiDAR autoencoding, order-invariant latent representations, and direct radar spectrum conditioning. These insights lead to a more compact and expressive generation process. Experiments show that RaLD produces dense and accurate 3D point clouds from raw radar spectrums, offering a promising solution for robust perception in challenging environments.

## Installation
Please refer to [Installation.md](doc/Installation.md) for installation instructions.

## Usage

### Data Preprocessing

The coloradar dataset can be downloaded from [coloradar dataset](https://arpg.github.io/coloradar/). Please follow the instructions on their website to get access to the dataset.

Move `resource/sequences_idx.tar.gz` and `split_files.tar.gz` to the root directory of ColoRadar dataset and extract it. The former provides the aligned radar index for each LiDAR frame and the latter provides the train/val/test split files.
  
Preprocess the dataset: 
  - Replace the `root_dir` and `output_dir` in `dataset_preprocessor/config/coloradar_config.yaml` and `dataset_preprocessor/config/coloradar_config_test_set.yaml` accordingly.
  
  - Running preprocessing for LiDAR:
    ```bash
    python dataset_preprocessor/lidar.py
    ```

  - Running preprocessing for Radar:
    ```bash
    python dataset_preprocessor/radar.py --mode sc
    ```

  - Cache the CFAR points in test set for evaluation:
    ```bash
    # cache RAE cube with higher resolution for CFAR processing
    python dataset_preprocessor/radar_test_set.py

    # cache the CFAR points for decoding process
    python dataset_preprocessor/cache_test_cfar.py --mode sc
    ```

### Training and Evaluation

Before running the training scripts, please make sure to adjust the configuration files in the `configs/` directory according to your setup, mainly the dataset paths and checkpoint saving paths.

- Auto-encoder training & evaluation: 

```
scripts/dist_train_ae.sh
``` 

- Generation model training & evaluation: 

```
scripts/dist_train_generation.sh
```

You can change the config file in the scripts to train/eval with different settings.

## Pretrained Models
You can download our pretrained models from [Google Drive](https://drive.google.com/drive/folders/1cR4kiaYV2iK59FPk3ECuCDZVb80oH0R2?usp=sharing). Models for auto-encoder and generation model are provided.

## Citation
If you find this code useful for your research, please consider citing the following paper:
```bibtex
@article{zhang2025rald,
  title={RaLD: Generating High-Resolution 3D Radar Point Clouds with Latent Diffusion},
  author={Zhang, Ruijie and Zeng, Bixin and Wang, Shengpeng and Zhou, Fuhui and Wang, Wei},
  journal={arXiv preprint arXiv:2511.07067},
  year={2025}
}

```

## Acknowledgements
We sincerely thank the following open-source projects for their great work, which has significantly facilitated our research:
- [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet)
- [MAR](https://github.com/LTH14/mar)
- [DiT](https://github.com/facebookresearch/DiT)
- [ColoRadar Toolkit](https://github.com/azinke/coloradar)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
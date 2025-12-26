## Installation

### 1. Set up a Python environment
You can use `conda` or `virtualenv` to create a new Python environment.
For example, using `conda`:
```bash
conda create -n RaLD python==3.12
conda activate RaLD
```
### 2. Install required packages
Install the required packages using `pip`:
```bash
pip install -r requirements.txt
``` 

### 3. Install additional dependencies
Install PyTorch with CUDA 12.4 support:
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
Install spconv with CUDA 12.4 support:
```bash
pip install spconv-cu124==2.3.8
pip install timm
pip install einops
pip install torch_cluster
pip install pandas
pip install natsort
```
Install pcdet v0.5:
> NOTE: Please re-install pcdet v0.5 by running python setup.py develop even if you have already installed previous version.
```bash
git clone https://github.com/open-mmlab/pcdet.git
cd pcdet
python setup.py develop
```
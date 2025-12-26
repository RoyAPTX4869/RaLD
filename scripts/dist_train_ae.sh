# !/bin/bash
cd $pwd/..

export CUDA_VISIBLE_DEVICES=0,1
 
# if you want to eval the model, just change config file to version for eval

config=configs/ae/ae_indoor_cfg_aniso_mix_view_cone.yml
n_gpus=2

main_script=main_ae.py

echo "running $main_script with config $config"

torchrun \
    --nproc_per_node=$n_gpus \
    --nnodes=1 \
    --master_port=29500 \
    $main_script \
    --config $config \
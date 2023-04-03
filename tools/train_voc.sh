#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

config=configs/pascal.yaml
save_path=exp/pascal/point

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    train_voc.py \
    --config=$config\
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.txt
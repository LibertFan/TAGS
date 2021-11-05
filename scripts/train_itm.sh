#!/usr/bin/env bash

ngpu=1
db_root=${db_root} # ../itm-data
txt_dir=${db_root}/text_db
img_dir=${db_root}/img_db
model_dir=${db_root}/pretrained
config_dir=.
zs_itm_result=./log
mkdir -p ${zs_itm_result}

horovodrun -np $ngpu python inf_itm.py \
    --txt_db ${txt_dir}/itm_flickr30k_test.db --img_db ${img_dir}/img/flickr30k \
    --checkpoint ${model_dir}/uniter-base.pt --model_config ${config_dir}/config/uniter-base.json \
    --output_dir ${zs_itm_result} --fp16 --pin_mem
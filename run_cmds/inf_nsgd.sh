#!/usr/bin/env bash

export NGPU=8
export DATA_ROOT=./UNITER/itm-data

export model_version=1
export arch=2000
export MODEL_ROOT=/mnt/Projects/UNITER/log/itm_nsgd/uniter_nsgd_base_v${model_version}/ckpt
export OUTPUT_DIR=./model_log/uniter_nsgd_base_v${model_version}/step${arch}
mkdir -p ${OUTPUT_DIR}

horovodrun -np $NGPU python inf_nsgd.py \
    --txt_db ${DATA_ROOT}/txt_db3/itm_flickr30k_test.db --img_db ${DATA_ROOT}/img_db/flickr30k \
    --checkpoint ${MODEL_ROOT}/model_step_${arch}.pt --model_config ./config/uniter-base.json \
    --output_dir ${OUTPUT_DIR} --fp16 --pin_mem

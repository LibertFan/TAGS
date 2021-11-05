#!/usr/bin/env bash

pwd

export ROOT_DIR=.

export NGPU=8
export LR=5e-5
export NSGD_SAMPLE_TEMPERATURE=2.0
export NSGD_RANK_LAMBDA=0.0005
export MLM_LAMBDA=0.05
export OUTPUT_DIR=${ROOT_DIR}/log/filckr/itm_pnsgd_base_v1

export NEGATIVE_SIZE=399
export HARD_NEG_SIZE=31
export MLM_SAMPLE_SIZE=20
export NSGD_SAMPLE_SIZE=20
export TRAIN_BATCH_SIZE=16
export MARGIN=0.2

export VALID_STEPS=500
export NUM_TRAIN_STEPS=5000
export WARMUP_STEPS=500

rm ${OUTPUT_DIR} -rf
ls -lh ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}

horovodrun -np ${NGPU} python train_pnsgd.py --config config/train-itm-pnsgd-base-flickr.json \
    --output_dir ${OUTPUT_DIR} --learning_rate ${LR} --negative_size ${NEGATIVE_SIZE} \
    --hard_neg_size ${HARD_NEG_SIZE} --mlm_sample_size ${MLM_SAMPLE_SIZE} --nsgd_sample_size ${NSGD_SAMPLE_SIZE} \
    --nsgd_sample_temperature ${NSGD_SAMPLE_TEMPERATURE} --train_batch_size ${TRAIN_BATCH_SIZE} \
    --mlm_lambda ${MLM_LAMBDA} --nsgd_rank_lambda ${NSGD_RANK_LAMBDA} --margin ${MARGIN} \
| tee -a ${OUTPUT_DIR}/train_log.txt

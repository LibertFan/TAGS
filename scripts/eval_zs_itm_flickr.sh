#!/usr/bin/env bash

horovodrun -np $NGPU python inf_itm.py \
    --txt_db ${TXT_DB} --img_db ${IMG_DB} --checkpoint ${PRETRAINED_MODEL} \
    --model_config ${MODEL_CONFIG} --output_dir $ZS_ITM_RESULT --fp16 --pin_mem
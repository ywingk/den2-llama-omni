#!/bin/bash

run_path="/raid/kyi/llama/wntg-LLaMA-Omni"

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="2,3"

# ------------------------------------------------
run_topic="eng_librispeech_train"

# ------------------------------------------------
question_file="kyi/questions/${run_topic}.json"
answer_file="kyi/answer/${run_topic}.json"
output_dir="${run_path}/kyi/expr/${run_topic}"
# ------------------------------------------------

cd ${run_path}

python omni_speech/train/stage1.py \
    --model-path models/Llama-3.2-1B-Instruct \
    --question-file ${question_file} \
    --answer-file ${answer_file} \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode llama_3 \
    --input_type mel \
    --mel_size 128 \
    --batch_size 16 \
    --accume_grad 4 \
    --train_epoch 5 \
    --output_dir ${output_dir}

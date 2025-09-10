#!/bin/bash

run_path="/home/kyi/den2-llama-omni"

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="2,3"

# ------------------------------------------------
run_topic="eng_LibriSpeech_train"

# ------------------------------------------------
question_file="questions/${run_topic}.json"
answer_file="answer/${run_topic}.json"
output_dir="${run_path}/expr/${run_topic}"
# ------------------------------------------------

cd ${run_path}
if [ ! -d "$output_dir" ]; then
  mkdir -p $output_dir
fi

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

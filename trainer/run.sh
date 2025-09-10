#!/bin/bash

run_path="/home/kyi/den2-llama-omni"

## =======================
if [ $# -eq 2 ]; then 
  dataset=$1
  task=$2
else
  echo; 
  echo " * Usage: bash $0 <dataset> <task> "
  echo " * <dataset>: kor/KsponSpeech, eng/LibriSpeech, ..."
  echo " * <task>: train, test "
  exit;
fi

# ------------------------------------------------
runTitle="${dataset/\//_}"
output_dir="${run_path}/expr/${runTitle}"
if [ ! -d "$output_dir" ]; then
  mkdir -p $output_dir
fi
# ------------------------------------------------

cd ${run_path}

if [ $task == "train" ]; then
  
  export CUDA_VISIBLE_DEVICES="1,2"

  question_file="questions/${runTitle}_train.json"
  answer_file="${output_dir}/answer_train.json"

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

elif [ $task == "test" ]; then

  export CUDA_VISIBLE_DEVICES="0"

  checkpoint="${expr_dir}/checkpoint-199000"
  question_file="questions/${runTitle}_test.json"
  answer_file="${output_dir}/answer_test.json"
  
  python omni_speech/infer/infer.py \
      --model-path ${checkpoint} \
      --question-file ${question_file} \
      --answer-file ${answer_file} \
      --num-chunks 1 \
      --chunk-idx 0 \
      --temperature 0 \
      --conv-mode llama_3 \
      --input_type mel \
      --mel_size 128 
fi

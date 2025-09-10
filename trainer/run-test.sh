#!/bin/bash

run_path="/home/kyi/den2-llama-omni"

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="1,2"

# ------------------------------------------------

test_data="eng_LibriSpeech_test"

# ------------------------------------------------
_dir="${run_path}/expr/${test_data}"
checkpoint_path="${expr_dir}/checkpoint-199000"
question_file="questions/${test_data}.json"
answer_file="answer/${test_data}.json"
# ------------------------------------------------

cd ${run_path}

python omni_speech/infer/infer.py \
    --model-path ${checkpoint_path} \
    --question-file ${question_file} \
    --answer-file ${answer_file} \
    --num-chunks 1 \
    --chunk-idx 0 \
    --temperature 0 \
    --conv-mode llama_3 \
    --input_type mel \
    --mel_size 128 

#    --s2s
#python omni_speech/infer/convert_jsonl_to_txt.py $ROOT/answer.json $ROOT/answer.unit
#python fairseq/examples/speech_to_speech/generate_waveform_from_code.py \
#    --in-code-file $ROOT/answer.unit \
#    --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
#    --results-path $ROOT/answer_wav/ --dur-prediction

# bash omni_speech/infer/run.sh omni_speech/infer/examples

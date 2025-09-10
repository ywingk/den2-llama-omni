#!/bin/bash

#speechdb_dir="/home/kyi/dcai-speechdb/eng/librispeech"
speechdb_dir="/mls/speechdb/eng/LibriSpeech"
kaldi_dir="${speechdb_dir}/kaldi"

dataset="dev train"
# ----------------------------------------
[ -f .done ] && echo "$0: done ==> Skip" && exit 0;
echo "$0: run data preparation - $scpdir"

for part in ${dataset}; do
    echo " - ${part} -"
    if [ ! -d ${part} ]; then 
        mkdir -p ./kaldi/${part}
    fi

    cp -f ${kaldi_dir}/${part}/wav.scp ./kaldi/${part}/
    cp -f ${kaldi_dir}/${part}/text.ipa ./kaldi/${part}/
    cp -f ${kaldi_dir}/${part}/text.spel ./kaldi/${part}/
done

touch .done && exit 0;
# -------------------------------------------



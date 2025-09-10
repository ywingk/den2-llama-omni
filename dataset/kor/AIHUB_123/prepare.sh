#!/bin/bash
#AIHUB_123 is the new name for Ksponspeech
kaldi_dir="/nas/speechdb/kor/AIHUB_123/kaldi_250411"
dataset="dev train"
# ----------------------------------------
[ -f .done ] && echo "$0: done ==> Skip" && exit 0;
echo "$0: run data preparation - $scpdir"

for part in ${dataset}; do
    echo " - ${part} -"
    if [ ! -d ${part} ]; then 
        mkdir ${part}
    fi

    cp -f ${kaldi_dir}/${part}/wav.scp ./${part}/
    #cp -f ${kaldi_dir}/${part}/text.ipa ./${part}/
    cp -f ${kaldi_dir}/${part}/text.pron ./${part}/
    cp -f ${kaldi_dir}/${part}/text.spel ./${part}/

    python3 ../../../utils/run-g2p.py ko \
        ./${part}/text.pron \
        ./${part}/text.ipa

    #python3 $tools_dir/make_raw_list.py ${part}/wav.scp \
    #  ${part}/text.ipa \
    #  ${part}/data.list.ipa

done

touch .done && exit 0;
# -------------------------------------------



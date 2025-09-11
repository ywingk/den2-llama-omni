#
import os, sys
import json

base_dir = "/home/kyi/den2-llama-omni"

# -----------------------------------------------------------------
if len(sys.argv) == 3:
    dataset_id, task = sys.argv[1:]
else:
    print(f" ** Usage: {sys.argv[0]} <dataset> <task> ")
    print(f" ** <dataset>: kor/AIHUB_123, eng/LibriSpeech, etc. ")
    print(f" ** <task>: train, test ")
    exit()

assert task in ["train", "test"], f"ERROR - check task {task}"
# -----------------------------------------------------------------

kaldi_path= os.path.join(base_dir, "corpus", dataset_id, "kaldi", task)
dataset_name = dataset_id.replace('/', '_')
output_dir = os.path.join(base_dir, "questions")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
questions_fn = os.path.join(output_dir, dataset_name+"_"+task+".json")

txt_fn = os.path.join(kaldi_path, 'text.spel')
scp_fn = os.path.join(kaldi_path, 'wav.scp')

with open(txt_fn, 'r') as tfp, open(scp_fn, 'r') as sfp:
    txt_lines = tfp.readlines()
    scp_lines = sfp.readlines()
#import pdb; pdb.set_trace()
assert len(txt_lines) == len(scp_lines), "check kaldi files"

with open(questions_fn, "w") as ofp:
    saved_array = []
    for idx, tline in enumerate(txt_lines):
        parts = tline.strip().split()
        key = parts[0]
        text = ' '.join(parts[1:])
        sline = scp_lines[idx]
        parts = sline.strip().split()
        assert key == parts[0], f"check key {key}"
        wav_fn = parts[1]
        #import pdb; pdb.set_trace()
        #print(f'- {key} {wav_fn}')
        data={
            "id":key,
            "speech":wav_fn,
            "conversations":[
                {
                    "from": "human",
                    "value": "<speech>\nPlease transcribe the speech accurately."
                },
                {
                    "from": "assistant",
                    "value": text
                }
            ]
        }
        saved_array.append(data)
    json.dump(saved_array, ofp, indent=4, ensure_ascii=False)


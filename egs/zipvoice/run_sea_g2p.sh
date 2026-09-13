#!/bin/bash

# This script is an example of training ZipVoice with SEA-G2P (Vietnamese/Thai/Indonesian)
# from scratch or fine-tuning on custom datasets.

# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=6

# Language for SEA-G2P: vi, th, id
lang=vi
tokenizer=sea_g2p

# Number of jobs for data preparation
nj=8

# Duration statistics (adjust according to your dataset)
train_hours=50
max_len=20

# Input data expects TSV files:
# data/raw/custom_train.tsv and data/raw/custom_dev.tsv
# Format: {uniq_id}\t{text}\t{wav_path}
for subset in train dev; do
      file_path=data/raw/custom_${subset}.tsv
      [ -f "$file_path" ] || { echo "Error: expect $file_path !" >&2; exit 1; }
done

### Stage 1: Prepare manifests from TSV files
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Prepare manifests for custom dataset from TSV files"

      for subset in train dev; do
            python3 -m zipvoice.bin.prepare_dataset \
                  --tsv-path data/raw/custom_${subset}.tsv \
                  --prefix custom \
                  --subset ${subset} \
                  --num-jobs ${nj} \
                  --output-dir data/manifests
      done
      # Output: data/manifests/custom_cuts_train.jsonl.gz, data/manifests/custom_cuts_dev.jsonl.gz
fi

### Stage 2: Prepare tokens file and pre-tokenize manifests
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Prepare tokens file and pre-tokenize manifests with SEA-G2P"

      python3 ./local/prepare_token_file_sea_g2p.py \
            --manifest data/manifests/custom_cuts_train.jsonl.gz \
            --tokens data/tokens_sea_g2p.txt \
            --lang ${lang}

      # Pre-tokenize manifests for much faster training
      for subset in train dev; do
            python3 -m zipvoice.bin.prepare_tokens \
                  --input-file data/manifests/custom_cuts_${subset}.jsonl.gz \
                  --output-file data/manifests/custom_cuts_${subset}_tokenized.jsonl.gz \
                  --tokenizer ${tokenizer} \
                  --lang ${lang} \
                  --num-jobs ${nj}
      done
fi

### Stage 3: Compute Fbank
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
      echo "Stage 3: Compute Fbank features for custom dataset"
      for subset in train dev; do
            python3 -m zipvoice.bin.compute_fbank \
                  --source-dir data/manifests \
                  --dest-dir data/fbank \
                  --dataset custom \
                  --subset ${subset} \
                  --num-jobs ${nj}
      done
fi

### Stage 4: Train ZipVoice model
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Train ZipVoice model with SEA-G2P"

      [ -z "$train_hours" ] && { echo "Error: train_hours is not set!" >&2; exit 1; }
      [ -z "$max_len" ] && { echo "Error: max_len is not set!" >&2; exit 1; }

      lr_hours=$(python3 -c "print(round(1000 * ($train_hours ** 0.3)))" )

      python3 -m zipvoice.bin.train_zipvoice \
            --world-size 1 \
            --use-fp16 1 \
            --num-iters 60000 \
            --max-duration 500 \
            --lr-hours ${lr_hours} \
            --max-len ${max_len} \
            --model-config conf/zipvoice_base.json \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --token-file data/tokens_sea_g2p.txt \
            --dataset custom \
            --train-manifest data/fbank/custom_cuts_train.jsonl.gz \
            --dev-manifest data/fbank/custom_cuts_dev.jsonl.gz \
            --exp-dir exp/zipvoice_${lang}
fi

### Stage 5: Average checkpoints
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
      echo "Stage 5: Average checkpoints"
      python3 -m zipvoice.bin.generate_averaged_model \
            --iter 60000 \
            --avg 2 \
            --model-name zipvoice \
            --exp-dir exp/zipvoice_${lang}
fi

### Stage 6: Inference
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
      echo "Stage 6: Inference test"
      python3 -m zipvoice.bin.infer_zipvoice \
            --model-name zipvoice \
            --model-dir exp/zipvoice_${lang} \
            --checkpoint-name iter-60000-avg-2.pt \
            --tokenizer ${tokenizer} \
            --lang ${lang} \
            --test-list test.tsv \
            --res-dir results/test_${lang}
fi

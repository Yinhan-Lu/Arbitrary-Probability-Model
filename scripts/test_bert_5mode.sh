#!/bin/bash

# Test BERT 5-mode evaluation with small dataset
# This runs a quick sanity check to verify all 5 evaluation modes work correctly

# Activate virtual environment
source myenv/bin/activate

python train.py \
  --model_type distilbert \
  --model_config distilbert-base-uncased \
  --dataset_name wikitext \
  --dataset_config wikitext-103-v1 \
  --num_epochs 3 \
  --num_train_samples 600 \
  --num_eval_samples 200 \
  --batch_size 1 \
  --eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --logging_steps 5 \
  --eval_steps 5 \
  --save_steps 999999 \
  --do_eval \
  --num_workers 0 \
  --device mps \
  --output_dir ./experiments \
  --exp_name mps_bert_full_dataset_script_test

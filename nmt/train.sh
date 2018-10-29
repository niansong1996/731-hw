#!/bin/sh

vocab="vocab.bin"
train_src="train.en-az.az.txt"
train_tgt="train.en-az.en.txt"
dev_src="../multilingual/data/dev.en-az.az.txt"
dev_tgt="../multilingual/data/dev.en-az.en.txt"
test_src="../multilingual/data/test.en-az.az.txt"
test_tgt="../multilingual/data/test.en-az.en.txt"

work_dir="work_dir"
name_prefix="embed"
model_name=${name_prefix}"-model.bin"
decode=${name_prefix}"-result.txt"
test_tgt="data/test.en-az.en.txt"
mkdir -p ${work_dir}
echo save results to ${work_dir}

python nmt.py \
    train \
    --cuda \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir}/${model_name} \
    --save-opt ${work_dir}/optimizer.bin \
    --valid-niter 1200 \
    --lr 0.001 \
    --log-every 50 \
    --batch-size 128 \
    --hidden-size 256 \
    --max-epoch 100 \
    --embed-size 256 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5
# 2>${work_dir}/err.log

python nmt.py \
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/${model_name} \
    ${test_src} \
    ${work_dir}/${decode}

perl multi-bleu.perl ${test_tgt} < ${work_dir}/${decode}

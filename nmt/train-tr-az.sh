#!/bin/sh

train_src="../multilingual/data/train.tr-az.tr.txt"
train_tgt="../multilingual/data/train.tr-az.az.txt"
dev_src="../multilingual/data/dev.tr-az.tr.txt"
dev_tgt="../multilingual/data/dev.tr-az.az.txt"
test_src="../multilingual/data/test.tr-az.tr.txt"
test_tgt="../multilingual/data/test.tr-az.az.txt"

work_dir="work_dir"
name_prefix="tr-az"
model_name=${name_prefix}"-model.bin"
decode=${name_prefix}"-result.txt"
mkdir -p ${work_dir}
echo save results to ${work_dir}

python nmt.py \
    train \
    --cuda \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --save-to ${work_dir}/${model_name} \
    --save-opt ${work_dir}/optimizer.bin \
    --valid-niter 1000 \
    --lr 0.001 \
    --log-every 50 \
    --batch-size 8 \
    --vocab-size 1000 \
    --hidden-size 128 \
    --max-epoch 100 \
    --embed-size 16 \
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

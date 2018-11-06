#!/bin/sh

vocab="data/vocab.bin"

work_dir="work_dir"
name_prefix="multi-lang"
model_name=${name_prefix}"-model.bin"
decode=${name_prefix}"-result.txt"
test_tgt="data/test.en-az.en.txt"
mkdir -p ${work_dir}
echo save results to ${work_dir}

python nmt.py \
    train \
    --langs 'az-en,tr-en'\
    --lang-embed-size 8\
    --cuda \
    --vocab-size 20000 \
    --save-to ${work_dir}/${model_name} \
    --save-opt ${work_dir}/optimizer.bin \
    --valid-niter 5 \
    --lr 0.001 \
    --log-every 50 \
    --batch-size 1 \
    --hidden-size 16 \
    --low-rank 2 \
    --num-layers 1 \
    --max-epoch 100 \
    --embed-size 300 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 \
    --patience 3
# 2>${work_dir}/err.log

python nmt.py \
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/${model_name} \
    az \
    en \
    ${work_dir}/${decode}

perl multi-bleu.perl ${test_tgt} < ${work_dir}/${decode}

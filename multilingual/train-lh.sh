#!/bin/sh

vocab="data/vocab.bin"

work_dir="work_dir"
name_prefix="multi-lang"
model_name=${name_prefix}"-model.bin"
decode=${name_prefix}"-result.txt"
mkdir -p ${work_dir}
echo save results to ${work_dir}

python nmt.py \
    train \
    --langs 'az-en,be-en,gl-en,tr-en,ru-en,pt-en,az-tr,be-ru,gl-pt'\
    --lang-embed-size 8\
    --cuda \
    --vocab-size 20000 \
    --save-to ${work_dir}/${model_name} \
    --save-opt ${work_dir}/optimizer.bin \
    --valid-niter 1000 \
    --lr 0.001 \
    --log-every 50 \
    --batch-size 32 \
    --hidden-size 256 \
    --low-rank 3 \
    --num-layers 1 \
    --max-epoch 100 \
    --embed-size 256 \
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
    decode-az.txt

python nmt.py \
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/${model_name} \
    be \
    en \
    decode-be.txt

python nmt.py \
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/${model_name} \
    gl \
    en \
    decode-gl.txt


perl multi-bleu.perl data/test.az-en.en.txt < decode-az.txt
perl multi-bleu.perl data/test.be-en.en.txt < decode-be.txt
perl multi-bleu.perl data/test.gl-en.en.txt < decode-gl.txt

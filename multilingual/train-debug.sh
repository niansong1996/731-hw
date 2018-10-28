#!/bin/sh

bash get_data.sh

work_dir="work_dir"
name_prefix="embed"
model_name=${name_prefix}"-model.bin"
decode=${name_prefix}"-result.txt"
test_tgt="data/test.en-az.en.txt"
mkdir -p ${work_dir}
echo save results to ${work_dir}

python align_vec.py 'az-tr,be-ru,gl-pt'

python nmt.py \
    train \
    --langs 'az-en,tr-en,be-en'\
    --lang-embed-size 8\
    --cuda \
    --vocab-size 25000 \
    --save-to ${work_dir}/${model_name} \
    --save-opt ${work_dir}/optimizer.bin \
    --valid-niter 2 \
    --lr 0.001 \
    --log-every 50 \
    --batch-size 8 \
    --hidden-size 64 \
    --low-rank 3 \
    --num-layers 1 \
    --max-epoch 100 \
    --embed-size 300 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 \
    --patience 3
# 2>${work_dir}/err.log

src="az"
tgt="en"

python nmt.py \
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/${model_name} \
    az \
    en \
    ${work_dir}/${src}"-"${tgt}"-"${decode}

perl multi-bleu.perl ${test_tgt} < ${work_dir}/${decode}

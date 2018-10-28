#!/bin/sh

bash get_data.sh

work_dir="work_dir"
name_prefix="embed"
model_name=${name_prefix}"-model.bin"
decode=${name_prefix}"-result.txt"
src="az"
tgt="en"
test_tgt="data/test.en-${src}.en.txt"
mkdir -p ${work_dir}
echo save results to ${work_dir}

python align_vec.py 'az-tr,be-ru,gl-pt'

python nmt.py \
    train \
    --langs 'az-en,be-en,gl-en,tr-en,ru-en,pt-en'\
    --lang-embed-size 8\
    --cuda \
    --vocab-size 10000 \
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
    ${src} \
    ${tgt} \
    ${work_dir}/${src}"-"${tgt}"-"${decode}

perl multi-bleu.perl ${test_tgt} < ${work_dir}/${src}"-"${tgt}"-"${decode}

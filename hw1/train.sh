#!/bin/sh

vocab="data/vocab.bin"
train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

work_dir="work_dir"
model_name="model-drop-embed-dict-hidden.bin"
decode="drop-embed-dict-hidden.txt"
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
    --hidden-size 512 \
    --max-epoch 100 \
    --embed-size 300 \
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

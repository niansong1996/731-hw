#!/bin/sh

vocab="vocab.bin"
train_src="train.all-en.all.txt"
train_tgt="train.all-en.en.txt"
dev_src="../multilingual/data/dev.all-en.all.txt"
dev_tgt="../multilingual/data/dev.all-en.en.txt"


work_dir="work_dir"
name_prefix="share-dec-enc-all-pair"
model_name=${name_prefix}"-model.bin"
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



langs='az gl be'
for lang in $langs
do
test_src="../multilingual/data/test.en-${lang}.${lang}.txt"
test_tgt="../multilingual/data/test.en-${lang}.${lang}.txt"
decode=${lang}"-result.txt"
python nmt.py \
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/${model_name} \
    ${test_src} \
    ${work_dir}/${decode}
perl multi-bleu.perl ${test_tgt} < ${work_dir}/${decode}
done




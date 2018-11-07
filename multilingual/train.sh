#!/bin/sh
echo "====== Current git branch: `git branch | grep \* | cut -d ' ' -f2`"
echo "====== Latest git commit:"
git show --summary
echo "====== Git diff:"
git diff
echo "====== current train.sh:"
cat train.sh
echo "=================="
work_dir="work_dir"
name_prefix="gl-pt-fasttext-load"
model_name=${name_prefix}"-model.bin"
decode=${name_prefix}"-result.txt"
mkdir -p ${work_dir}
echo save results to ${work_dir}

python nmt.py \
    train \
    --langs 'gl-en,pt-en'\
    --lang-embed-size 8\
    --cuda \
    --vocab-size 20000 \
    --save-to ${work_dir}/${model_name} \
    --save-opt ${work_dir}/optimizer.bin \
    --valid-niter 1000 \
    --lr 0.001 \
    --log-every 50 \
    --batch-size 64 \
    --hidden-size 512 \
    --low-rank 3 \
    --num-layers 2 \
    --max-epoch 100 \
    --embed-size 200 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --denoising 0.2 \
    --clip-grad 5.0 \
    --autoencode-epoch 5 \
    --lr-decay 0.5 \
    --patience 5 \
    --max-num-trial 1
 2>${work_dir}/err.log

for lang in pt 
do
python nmt.py decode --cuda --beam-size 5 --max-decoding-time-step 100 \
    ${work_dir}/${model_name} ${lang} en ${work_dir}/decode-${lang}.txt
perl multi-bleu.perl data/test.${lang}-en.en.txt < ${work_dir}/decode-${lang}.txt
done

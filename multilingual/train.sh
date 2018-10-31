#!/bin/sh

work_dir="work_dir"
name_prefix="multi-lang"
model_name=${name_prefix}"-model.bin"
decode=${name_prefix}"-result.txt"
mkdir -p ${work_dir}
echo save results to ${work_dir}

# 2>${work_dir}/err.log

python nmt.py decode --cuda --beam-size 5 --max-decoding-time-step 100 \
    ${work_dir}/${model_name} de en ${work_dir}/decode-de.txt
perl multi-bleu.perl data/test.de-en.en.txt < ${work_dir}/decode-de.txt

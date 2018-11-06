#!/usr/bin/env bash
lang=$1
train_ted="multilingual/data/train.${lang}-en.${lang}.txt"
train_ted_prep="${train_ted}.prep"
dev_ted="multilingual/data/dev.${lang}-en.${lang}.txt"
dev_ted_prep="${dev_ted}.prep"
wiki_file="multilingual/data/${lang}.wiki.txt"
wiki_prep="${wiki_file}.prep"
mono="multilingual/data/${lang}.mono.txt"
python multilingual/wiki_prep.py --lower-case=True --min-len=1 --max-len=80 --max-size=2000000 ${train_ted}
python multilingual/wiki_prep.py --lower-case=True --min-len=1 --max-len=80 --max-size=2000000 ${dev_ted}
train_lines=`wc -l < "${train_ted_prep}"`
dev_lines=`wc -l < "${dev_ted_prep}"`
ted_lines=$(($train_lines+$dev_lines))
sample_wiki_lines=$((10 * ${ted_lines}))
python multilingual/wiki_prep.py --lower-case=True --min-len=5 --max-len=80 --max-size=${sample_wiki_lines} ${wiki_file}
wiki_lines=`wc -l < "${wiki_prep}"`
#if (($wiki_lines > 10 * ${ted_lines}))
#
#then
#    gshuf -n ${sample_wiki_lines} "${wiki_prep}" | cat "${train_ted_prep}" "${dev_ted_prep}" - > ${mono}
#    echo "sampled ${sample_wiki_lines} lines from wiki to get `wc -l < ${mono}` lines mono"
#else
#    cat "${train_ted_prep}" "${dev_ted_prep}" "${wiki_prep}" > ${mono}
#fi
cat "${train_ted_prep}" "${dev_ted_prep}" "${wiki_prep}" > ${mono}
../fastText/fastText skipgram -input ${mono} -output ~/Downloads/${lang}.embed -dim 200

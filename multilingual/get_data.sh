#!/bin/sh

# get raw data if it does not exist
if [ ! -d "./data" ]; then
    wget http://phontron.com/class/mtandseq2seq2018/assets/data/cs11731-assignment2-v1.zip
    unzip cs11731-assignment2-v1.zip
    unzip assignment-2/wikis.zip
    mkdir data
    cp wikis/* data/

    wget http://phontron.com/data/ted_talks.tar.gz
    tar xvf ted_talks.tar.gz
    python extract_ted_talks.py
fi

# combine monolingual sentences
cat data/train.az-en.en.txt \
    data/train.be-en.en.txt \
    data/train.gl-en.en.txt \
    data/train.tr-en.en.txt \
    data/train.ru-en.en.txt \
    data/train.pt-en.en.txt > data/en_mono.txt
cat data/train.az-en.az.txt data/train.tr-en.tr.txt > data/aztr_mono.txt
cat data/train.be-en.be.txt data/train.ru-en.ru.txt > data/beru_mono.txt
cat data/train.gl-en.gl.txt data/train.pt-en.pt.txt > data/glpt_mono.txt

# concat the low-high resource data
for set_type in "train" "test" "dev"
do
cat data/${set_type}.az-en.az.txt data/${set_type}.tr-en.tr.txt > data/${set_type}.aztr-en.aztr.txt
cat data/${set_type}.az-en.en.txt data/${set_type}.tr-en.en.txt > data/${set_type}.aztr-en.en.txt
cat data/${set_type}.be-en.be.txt data/${set_type}.ru-en.ru.txt > data/${set_type}.beru-en.beru.txt
cat data/${set_type}.be-en.en.txt data/${set_type}.ru-en.en.txt > data/${set_type}.beru-en.en.txt
cat data/${set_type}.gl-en.gl.txt data/${set_type}.pt-en.pt.txt > data/${set_type}.glpt-en.glpt.txt
cat data/${set_type}.gl-en.en.txt data/${set_type}.pt-en.en.txt > data/${set_type}.glpt-en.en.txt
done

# clean up
rm all_talks_train.tsv
rm all_talks_dev.tsv
rm all_talks_test.tsv
rm ted_talks.tar.gz
rm -rf wikis
rm -rf assignment-2
rm cs11731-assignment2-v1.zip







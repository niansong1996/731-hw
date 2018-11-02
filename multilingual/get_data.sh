#!/bin/sh

# get raw data if it does not exist
if [ ! -d "./data" ]; then
    wget http://phontron.com/class/mtandseq2seq2018/assets/data/cs11731-assignment2-v1.zip
    unzip cs11731-assignment2-v1.zip
    unzip assignment-2/wikis.zip
    mkdir data
    cp wikis/* data/

    # get raw wikipedia data
    cp assignment-2/wikipedia.sh ./
    wget https://github.com/attardi/wikiextractor/archive/master.zip
    unzip master.zip
    rm master.zip
    wget https://github.com/moses-smt/mosesdecoder/archive/master.zip
    unzip master.zip
    rm master.zip

    # get raw ted data
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
cat data/train.az-en.az.txt data/az.wiki.txt > data/az_mono.txt
cat data/train.be-en.be.txt data/be.wiki.txt > data/be_mono.txt
cat data/train.gl-en.gl.txt data/gl.wiki.txt > data/gl_mono.txt
cat data/train.tr-en.tr.txt > data/tr_mono.txt
cat data/train.ru-en.ru.txt > data/ru_mono.txt
cat data/train.pt-en.pt.txt > data/pt_mono.txt

# clean up
rm all_talks_train.tsv
rm all_talks_dev.tsv
rm all_talks_test.tsv
rm ted_talks.tar.gz
rm -rf wikis
rm -rf assignment-2
rm cs11731-assignment2-v1.zip







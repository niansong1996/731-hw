#!/bin/sh

# get raw data if it does not exist
if [ ! -d "./data" ]; then
    wget http://phontron.com/class/mtandseq2seq2018/assets/data/cs11731-assignment2-v1.zip
    unzip cs11731-assignment2-v1.zip
    unzip assignment-2/data.zip
    unzip assignment-2/wikis.zip
    cp wikis/* data/
fi


# combine monolingual sentences
cat data/train.en-az.en.txt \
    data/train.en-be.en.txt \
    data/train.en-gl.en.txt \
    data/train.en-tr.en.txt \
    data/train.en-ru.en.txt \
    data/train.en-pt.en.txt > data/en_mono.txt
cat data/train.en-az.az.txt data/az.wiki.txt > data/az_mono.txt
cat data/train.en-be.be.txt data/be.wiki.txt > data/be_mono.txt
cat data/train.en-gl.gl.txt data/gl.wiki.txt > data/gl_mono.txt
cat data/train.en-tr.tr.txt > data/tr_mono.txt
cat data/train.en-ru.ru.txt > data/ru_mono.txt
cat data/train.en-pt.pt.txt > data/pt_mono.txt

# clean up
rm -rf wikis
rm -rf assignment-2
rm cs11731-assignment2-v1.zip







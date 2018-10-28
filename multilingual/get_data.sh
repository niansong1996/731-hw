#!/bin/sh

# get raw data if it does not exist
if [ ! -d "./data" ]; then
    wget http://phontron.com/class/mtandseq2seq2018/assets/data/cs11731-assignment2-v1.zip
    unzip cs11731-assignment2-v1.zip
    unzip assignment-2/data.zip
fi


# combine monolingual sentences
cat data/train.en-az.az.txt data/train.en-tr.tr.txt > data/train.en-az.az.txt
cat data/train.en-az.en.txt data/train.en-tr.en.txt > data/train.en-az.en.txt

# clean up
rm -rf assignment-2
rm cs11731-assignment2-v1.zip







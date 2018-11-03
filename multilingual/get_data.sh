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

# concat the low-high resource data
for set_type in "train"
do
cat data/${set_type}.az-en.az.txt data/${set_type}.tr-en.tr.txt > data/${set_type}.aztr-en.aztr.txt
cat data/${set_type}.az-en.en.txt data/${set_type}.tr-en.en.txt > data/${set_type}.aztr-en.en.txt
cat data/${set_type}.be-en.be.txt data/${set_type}.ru-en.ru.txt > data/${set_type}.beru-en.beru.txt
cat data/${set_type}.be-en.en.txt data/${set_type}.ru-en.en.txt > data/${set_type}.beru-en.en.txt
cat data/${set_type}.gl-en.gl.txt data/${set_type}.pt-en.pt.txt > data/${set_type}.glpt-en.glpt.txt
cat data/${set_type}.gl-en.en.txt data/${set_type}.pt-en.en.txt > data/${set_type}.glpt-en.en.txt
done

# get test set using only az be gl
cat data/dev.az-en.en.txt > data/dev.aztr-en.en.txt
cat data/dev.az-en.az.txt > data/dev.aztr-en.aztr.txt
cat data/dev.be-en.en.txt > data/dev.beru-en.en.txt
cat data/dev.be-en.be.txt > data/dev.beru-en.beru.txt
cat data/dev.gl-en.en.txt > data/dev.glpt-en.en.txt
cat data/dev.gl-en.gl.txt > data/dev.glpt-en.glpt.txt

# get test set using only az be gl
cat data/test.az-en.en.txt > data/test.aztr-en.en.txt
cat data/test.az-en.az.txt > data/test.aztr-en.aztr.txt
cat data/test.be-en.en.txt > data/test.beru-en.en.txt
cat data/test.be-en.be.txt > data/test.beru-en.beru.txt
cat data/test.gl-en.en.txt > data/test.glpt-en.en.txt
cat data/test.gl-en.gl.txt > data/test.glpt-en.glpt.txt

# clean up
rm all_talks_train.tsv
rm all_talks_dev.tsv
rm all_talks_test.tsv
rm ted_talks.tar.gz
rm -rf wikis
rm -rf assignment-2
rm cs11731-assignment2-v1.zip







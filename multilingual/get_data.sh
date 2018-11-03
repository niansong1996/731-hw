#!/bin/sh

# get raw data if it does not exist
wget http://phontron.com/class/mtandseq2seq2018/assets/data/cs11731-assignment2-v1.zip
unzip cs11731-assignment2-v1.zip
unzip assignment-2/wikis.zip
mkdir data
cp wikis/* data/

# get raw ted data
wget http://phontron.com/data/ted_talks.tar.gz
tar xvf ted_talks.tar.gz
python extract_ted_talks.py

# clean up
rm all_talks_train.tsv
rm all_talks_dev.tsv
rm all_talks_test.tsv
rm ted_talks.tar.gz
rm -rf wikis
rm -rf assignment-2
rm cs11731-assignment2-v1.zip







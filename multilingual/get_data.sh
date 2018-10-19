#!/bin/sh

wget http://phontron.com/class/mtandseq2seq2018/assets/data/cs11731-assignment2-v1.zip
unzip cs11731-assignment2-v1.zip
unzip assignment-2/data.zip
unzip assignment-2/wikis.zip
cp wikis/* data/

# clean up
rm -rf wikis
rm -rf assignment-2
rm cs11731-assignment2-v1.zip







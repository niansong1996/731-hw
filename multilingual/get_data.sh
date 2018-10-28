#!/bin/sh

# get raw data if it does not exist
if [ ! -d "./data" ]; then
    wget http://phontron.com/class/mtandseq2seq2018/assets/data/cs11731-assignment2-v1.zip
    unzip cs11731-assignment2-v1.zip
    unzip assignment-2/data.zip
    # clean up
    rm -rf wikis
    rm -rf assignment-2
    rm cs11731-assignment2-v1.zip
fi

if [ ! -d "./embed" ]; then
    mkdir embed
    cd embed
    wget http://cosyne.h-its.org/bpemb/data/az/az.wiki.bpe.op10000.model
    wget http://cosyne.h-its.org/bpemb/data/az/az.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    tar -xvzf az.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    rm az.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    
    wget http://cosyne.h-its.org/bpemb/data/be/be.wiki.bpe.op10000.model
    wget http://cosyne.h-its.org/bpemb/data/be/be.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    tar -xvzf be.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    rm be.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    
    wget http://cosyne.h-its.org/bpemb/data/gl/gl.wiki.bpe.op10000.model
    wget http://cosyne.h-its.org/bpemb/data/gl/gl.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    tar -xvzf gl.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    rm gl.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    
    wget http://cosyne.h-its.org/bpemb/data/tr/tr.wiki.bpe.op10000.model
    wget http://cosyne.h-its.org/bpemb/data/tr/tr.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    tar -xvzf tr.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    rm tr.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    
    wget http://cosyne.h-its.org/bpemb/data/ru/ru.wiki.bpe.op10000.model
    wget http://cosyne.h-its.org/bpemb/data/ru/ru.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    tar -xvzf ru.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    rm ru.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    
    wget http://cosyne.h-its.org/bpemb/data/pt/pt.wiki.bpe.op10000.model
    wget http://cosyne.h-its.org/bpemb/data/pt/pt.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    tar -xvzf pt.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    rm pt.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    
    wget http://cosyne.h-its.org/bpemb/data/en/en.wiki.bpe.op10000.model
    wget http://cosyne.h-its.org/bpemb/data/en/en.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    tar -xvzf en.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    rm en.wiki.bpe.op10000.d300.w2v.txt.tar.gz
    cd ../
fi

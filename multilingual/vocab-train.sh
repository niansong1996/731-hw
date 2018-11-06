#!/usr/bin/env bash
mkdir -p vocab
python MultiMT.py --train-src=./data/az.mono.txt --size=20000 --freq-cutoff=2 vocab/vocab-az.bin
python MultiMT.py --train-src=./data/be.mono.txt --size=20000 --freq-cutoff=2 vocab/vocab-be.bin
python MultiMT.py --train-src=./data/gl.mono.txt --size=20000 --freq-cutoff=2 vocab/vocab-gl.bin
python MultiMT.py --train-src=./data/train.tr-en.tr.txt --size=20000 --freq-cutoff=2 vocab/vocab-tr.bin
python MultiMT.py --train-src=./data/train.ru-en.ru.txt --size=20000 --freq-cutoff=2 vocab/vocab-ru.bin
python MultiMT.py --train-src=./data/train.pt-en.pt.txt --size=20000 --freq-cutoff=2 vocab/vocab-pt.bin
python MultiMT.py --train-src=./data/train.en-en.en.txt --size=20000 --freq-cutoff=2 vocab/vocab-en.bin

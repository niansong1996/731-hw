# preprocess the wiki data
python wiki_prep.py data/az.wiki.txt subword_files/az.5000.txt
python wiki_prep.py data/be.wiki.txt subword_files/be.5000.txt
python wiki_prep.py data/gl.wiki.txt subword_files/gl.5000.txt
python wiki_prep.py data/en.wiki.txt subword_files/en.5000.txt

# combine monolingual sentences
cat data/train.az-en.en.txt \
    data/train.be-en.en.txt \
    data/train.gl-en.en.txt \
    data/train.tr-en.en.txt \
    data/train.ru-en.en.txt \
    data/train.pt-en.en.txt \
    data/en.wiki.txt.prep > data/en_mono.txt
cat data/train.az-en.az.txt data/az.wiki.txt.prep > data/az_mono.txt
cat data/train.be-en.be.txt data/be.wiki.txt.prep > data/be_mono.txt
cat data/train.gl-en.gl.txt data/gl.wiki.txt.prep > data/gl_mono.txt
cat data/train.tr-en.tr.txt > data/tr_mono.txt
cat data/train.ru-en.ru.txt > data/ru_mono.txt
cat data/train.pt-en.pt.txt > data/pt_mono.txt

# create autoencode data
cp data/en.wiki.txt.prep data/train.en-en.en.txt
cp data/az.wiki.txt.prep data/train.az-az.az.txt
cp data/be.wiki.txt.prep data/train.be-be.be.txt
cp data/gl.wiki.txt.prep data/train.gl-gl.gl.txt

cp data/test.az-en.az.txt data/dev.az-az.az.txt
cp data/test.be-en.be.txt data/dev.be-be.be.txt
cp data/test.gl-en.gl.txt data/dev.gl-gl.gl.txt
cp data/test.pt-en.en.txt data/dev.en-en.en.txt

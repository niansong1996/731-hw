# concat the low-high resource data
for set_type in "train" "dev"
do
cat data/${set_type}.az-en.az.txt data/${set_type}.tr-en.tr.txt > data/${set_type}.aztr-en.aztr.txt
cat data/${set_type}.az-en.en.txt data/${set_type}.tr-en.en.txt > data/${set_type}.aztr-en.en.txt
cat data/${set_type}.be-en.be.txt data/${set_type}.ru-en.ru.txt > data/${set_type}.beru-en.beru.txt
cat data/${set_type}.be-en.en.txt data/${set_type}.ru-en.en.txt > data/${set_type}.beru-en.en.txt
cat data/${set_type}.gl-en.gl.txt data/${set_type}.pt-en.pt.txt > data/${set_type}.glpt-en.glpt.txt
cat data/${set_type}.gl-en.en.txt data/${set_type}.pt-en.en.txt > data/${set_type}.glpt-en.en.txt
done

# get test set using only az be gl
cat data/test.az-en.en.txt > data/test.aztr-en.en.txt
cat data/test.az-en.az.txt > data/test.aztr-en.aztr.txt
cat data/test.be-en.en.txt > data/test.beru-en.en.txt
cat data/test.be-en.be.txt > data/test.beru-en.beru.txt
cat data/test.gl-en.en.txt > data/test.glpt-en.en.txt
cat data/test.gl-en.gl.txt > data/test.glpt-en.glpt.txt

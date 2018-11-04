# concat the low-high mono lingual data for subword training
cat data/az_mono.txt data/tr_mono.txt > data/aztr_mono.txt
cat data/be_mono.txt data/ru_mono.txt > data/beru_mono.txt
cat data/gl_mono.txt data/pt_mono.txt > data/glpt_mono.txt

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
for set_type in "dev" "test"
do
cat data/${set_type}.az-en.en.txt > data/${set_type}.aztr-en.en.txt
cat data/${set_type}.az-en.az.txt > data/${set_type}.aztr-en.aztr.txt
cat data/${set_type}.be-en.en.txt > data/${set_type}.beru-en.en.txt
cat data/${set_type}.be-en.be.txt > data/${set_type}.beru-en.beru.txt
cat data/${set_type}.gl-en.en.txt > data/${set_type}.glpt-en.en.txt
cat data/${set_type}.gl-en.gl.txt > data/${set_type}.glpt-en.glpt.txt
done

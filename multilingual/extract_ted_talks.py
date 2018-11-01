# coding: utf-8

"""
extract training, dev, test data for gl, az, be, pt, tr, ru
"""

import io
import csv
import sys
from builtins import str

ftrain = io.open('all_talks_train.tsv','r',encoding='utf-8')
fdev = io.open('all_talks_dev.tsv','r',encoding='utf-8')
ftest = io.open('all_talks_test.tsv','r',encoding='utf-8')

def get_language_pairs(src, tgt):
    def get_data(csv_f):
        data = []
        reader = csv.DictReader(csv_f, delimiter='\t')
        for row in reader:
            en, fr = row[src].strip(), row[tgt].strip()
            en = en.replace("__NULL__","").replace('_ _ NULL _ _','').strip()
            fr = fr.replace("__NULL__","").replace('_ _ NULL _ _','').strip()
            if len(en) == 0 or len(fr) == 0:
                continue
            data.append((en,fr))
        csv_f.seek(0)
        return data
    tr, de, ts = get_data(ftrain), get_data(fdev), get_data(ftest)
    
    def write_data(data, fname1, fname2):
        f1 = io.open(fname1,'w',encoding='utf-8')
        f2 = io.open(fname2,'w',encoding='utf-8')
        print(len(data))
        for i in data:
            f1.write(str(i[0])+"\n")
            f2.write(str(i[1])+"\n")
        f1.close()
        f2.close()

    write_data(tr, ("./data/train.%s-%s.%s.txt" % (src, tgt, src)), ("./data/train.%s-%s.%s.txt" % (src, tgt, tgt)) ) 
    write_data(de, ("./data/dev.%s-%s.%s.txt" % (src, tgt, src)), ("./data/dev.%s-%s.%s.txt" % (src, tgt, tgt)) ) 
    write_data(ts, ("./data/test.%s-%s.%s.txt" % (src, tgt, src)), ("./data/test.%s-%s.%s.txt" % (src, tgt, tgt)) ) 

# get_language_pairs("gl","en")
# get_language_pairs("az","en")
# get_language_pairs("be","en")
# 
# get_language_pairs("pt","en")
# get_language_pairs("tr","en")
# get_language_pairs("ru","en")
# 
# get_language_pairs("gl","pt")
# get_language_pairs("az","tr")
# get_language_pairs("be","ru")
get_language_pairs("tr", "az")

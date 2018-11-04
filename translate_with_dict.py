def lookup_dict(w, d, trans_cnt):
    if w in d:
        trans_cnt[0] = trans_cnt[0] + 1
        # return '>>>[{}<-{}]<<<'.format(d[w], w)
        return d[w]
    return w


def translate(input_file, output_file, d):
    line_num = 1
    trans_line = []
    with open(output_file, 'w') as f:
        for line in open(input_file, encoding="utf-8"):
            sent = line.strip().split(' ')
            trans_cnt = [0]
            translated_words = map(lambda w:  lookup_dict(w, d, trans_cnt), sent)
            trans_sent = ' '.join(translated_words)
            if '[' in trans_sent:
                trans_line.append(line_num)
            line_num += 1
            f.write(trans_sent + '\n')
    return trans_line


def get_vocab(file):
    vocab = {}
    for line in open(file, encoding="utf-8"):
        for w in line.strip().split(' '):
            if w in vocab:
                vocab[w] += 1
            else:
                vocab[w] = 0
    return vocab


def read_dict(file_path, src_vocab, tgt_vocab):
    d = {}
    line_count = 0
    ommitted_cnt = 0
    for line in open(file_path, encoding="utf-8"):
        entry = line.strip().split(' ')
        line_count += 1
        src = entry[0]
        tgt = entry[1]
        score = float(entry[2])
        if src == tgt:
            ommitted_cnt += 1
            continue
        if src not in src_vocab or src_vocab[src] < 3:
            continue
        if tgt not in tgt_vocab or tgt_vocab[tgt] < 3:
            continue
        if score < 0.6:
            break
        d[src] = tgt
    print("Got dict of size %d; omitted %d words with the same spelling"%(len(d), ommitted_cnt))
    return d

if __name__ == '__main__':
    # src_vocab = get_vocab("cleaned_tr-az.tr")
    # tgt_vocab = get_vocab("cleaned_tr-az.az")
    # d = read_dict('lex.tr2az.sorted', src_vocab, tgt_vocab)
    # lines = translate('multilingual/data/test.tr-az.tr.txt', 'test.result.tr-az.az.txt', d)
    # print(lines)

    src_vocab = get_vocab("cleaned-es-gl.es")
    tgt_vocab = get_vocab("cleaned-es-gl.gl")
    d = read_dict('lex.pt2gl.sorted', src_vocab, tgt_vocab)
    lines = translate('multilingual/data/train.es-gl.es.txt', 'train.result.es-gl.gl.txt', d)
    print(lines)
    
    # src_vocab = get_vocab("cleaned-ru-be.ru")
    # tgt_vocab = get_vocab("cleaned-ru-be.be")
    # d = read_dict('lex.ru2be.sorted', src_vocab, tgt_vocab)
    # lines = translate('multilingual/data/train.ru-be.ru.txt', 'train.result.ru-be.be.txt', d)
    # print(lines)

    # src_vocab = get_vocab("cleaned-uk-be.uk")
    # tgt_vocab = get_vocab("cleaned-uk-be.be")
    # d = read_dict('lex.ru2be.sorted', src_vocab, tgt_vocab)
    # lines = translate('multilingual/data/train.uk-be.uk.txt', 'train.result.uk-be.be.txt', d)
    # print(lines)
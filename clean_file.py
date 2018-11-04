def clean(src_file, tgt_file, src_output, tgt_output):
    src_lines = []
    for line in open(src_file, encoding="utf-8"):
        src_lines.append(line)
    tgt_lines = []
    for line in open(tgt_file, encoding="utf-8"):
        tgt_lines.append(line)
    src_result = []
    tgt_result = []
    for src, tgt in zip(src_lines, tgt_lines):
        len_ratio = len(src.split(" ")) / len(tgt.split(" "))
        if 0.8 < len_ratio < 1.2:
            src_result.append(src)
            tgt_result.append(tgt)
    with open(src_output, 'w') as f:
        for line in src_result:
            f.write(line)
    with open(tgt_output, 'w') as f:
        for line in tgt_result:
            f.write(line)
    print("reduce %d lines to %d"%(len(src_lines), len(src_result)))


if __name__ == "__main__":
    # clean("multilingual/data/train.ru-be.ru.txt", "multilingual/data/train.ru-be.be.txt",
    #       "cleaned-ru-be.ru", "cleaned-ru-be.be")
    # clean("multilingual/data/train.uk-be.uk.txt", "multilingual/data/train.uk-be.be.txt",
    #       "cleaned-uk-be.uk", "cleaned-uk-be.be")
    clean("multilingual/data/train.pt-gl.pt.txt", "multilingual/data/train.pt-gl.gl.txt",
          "cleaned-pt-gl.pt", "cleaned-pt-gl.gl")

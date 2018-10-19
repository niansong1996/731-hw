import sentencepiece as spm

def train():
    spm.SentencePieceTrainer.Train('--input=data/')
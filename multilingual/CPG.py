import torch
import torch.nn as nn
import torch.tensor as Tensor
import torch.nn.functional as F

LANG_INDICES = { 'en' : 0, 
                 'gl' : 1, 'pt' : 4,
                 'az' : 2, 'tr' : 5,
                 'be' : 3, 'ru' : 6 }


class CPG(nn.Module):
    def __init__(self, enc_shapes, dec_shapes, size_dict):
        super(CPG, self).__init__()

        # init size constants
        self.enc_shapes = enc_shapes
        self.dec_shapes = dec_shapes
        self.lang_embed_size = size_dict['lang_embed_size']
        self.word_embed_size = size_dict['word_embed_size']
        self.num_lang = size_dict['num_lang']
        self.vocab_size = size_dict['vocab_size']
        self.low_rank = size_dict['low_rank']
        self.lang_encode = torch.eyes(self.num_lang)

        # calculate the parameter number
        self.enc_param_num = sum([shape[0]*shape[1] for shape in enc_shapes])
        self.dec_param_num = sum([shape[0]*shape[1] for shape in dec_shapes])


        # init all the layers
        self.L = nn.Linear(self.num_lang, self.lang_embed_size, bias=False)
        self.enc_P = nn.Linear(self.lang_embed_size, self.low_rank)
        self.enc_W = nn.Linear(self.low_rank, self.enc_param_num)
        self.dec_P = nn.Linear(self.lang_embed_size, self.low_rank)
        self.dec_W = nn.Linear(self.low_rank, self.dec_param_num)

        # init language embeddings
        self.word_embeddings = []
        for _ in range(self.num_lang):
            self.word_embeddings.append(nn.Embedding(self.vocab_size, self.word_embed_size))
        

        # initialize the parameters using uniform distribution
        for param in self.parameters():
            nn.init.uniform_(param.data, a=-0.1, b=0.1)


    def get_params(self, lang: int, enc: bool):
        # get params for encoder or decoder
        ell = self.L(self.lang_encode[lang])
        P_ell = self.enc_P(ell) if enc else self.dec_P(ell)
        W_P_ell = self.enc_W(ell) if enc else self.dec_W(ell)
        theta = W_P_ell

        return theta


    def get_embedding(self, lang: int):
        # get the word embedding for the language
        word_embedding =  self.word_embeddings[lang]

        return word_embedding


    @DeprecationWarning
    def forward(self, L, X, y):
        pass 
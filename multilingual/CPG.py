from typing import *

import torch
import torch.nn as nn
import torch.tensor as Tensor
from functools import reduce
import os.path

from config import device, LANG_INDICES
import numpy as np
from config import LANG_NAMES
import fastText


class Embed:
    def __init__(self, lang, vocab, vocab_size, word_embed_size):
        weights_path = 'embed/%s.embed.npy' % lang
        self.lang = lang
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.word_embed_size = word_embed_size
        self.emb_layer = None
        if os.path.isfile(weights_path):
            print("loading npy model for %s" % lang)
            weights_matrix = np.load(weights_path)
            self.emb_layer = nn.Embedding(vocab_size, word_embed_size)
            self.emb_layer.weight = nn.Parameter(torch.from_numpy(weights_matrix).float())
            self.emb_layer.weight.requires_grad = False
            # print("Done loading npy model for %s" % lang)
        else:
            print("loading fastext model for %s" % lang)
            self.model = fastText.load_model('embed_bkp/%s.embed.bin' % lang)
            self.vocab = vocab
            print("Done loading fastext model for %s" % lang)

    def __getstate__(self):
        """Return state values to be pickled."""
        return (self.lang,  self.vocab, self.vocab_size, self.word_embed_size)

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        lang, vocab, vocab_size, word_embed_size = state
        self.__init__(lang, vocab, vocab_size, word_embed_size)

    def __call__(self, src_sent_idx: np.ndarray):
        if self.emb_layer:
            # print("getting embedding for %s from embed layer" % self.lang)
            return self.emb_layer(torch.tensor(src_sent_idx, dtype=torch.long)).to(device)
        else:
            # print("getting embedding for %s from fastText" % self.lang)
            word_vecs = [self.model.get_word_vector(self.vocab.id2word[idx])
                         for idx in src_sent_idx.reshape(-1)]
            embed_tensor = torch.tensor(word_vecs, dtype=torch.float32, device=device)
            if isinstance(src_sent_idx[0], np.ndarray):
                return embed_tensor.reshape(len(src_sent_idx), len(src_sent_idx[0]), -1)
            else:
                return embed_tensor.reshape(len(src_sent_idx), -1)


class CPG(nn.Module):
    def __init__(self, vocabs, shapes: List[List[Tuple[int]]], args: Dict[str, str], encoder_group: int):
        """
        Args:
            shapes: List[List[tuples]] a list of groups, where each tuple the
                    shape of the params in that group
        """
        super(CPG, self).__init__()

        # init size constants

        self.lang_embed_size = int(args['--lang-embed-size'])
        self.word_embed_size = int(args['--embed-size'])
        self.vocab_size = int(args['--vocab-size'])
        self.low_rank = int(args['--low-rank'])
        num_lang = len(LANG_NAMES)
        self.lang_encode = torch.eye(num_lang, device=device)

        self.shapes = shapes
        self.group_num, self.group_param_num, self.group_param_sizes = self.get_param_meta(shapes)

        # init every layer of CPG for different groups
        self.L = nn.Linear(num_lang, self.lang_embed_size, bias=False)
        self.Ps = nn.ModuleList([nn.Linear(self.lang_embed_size, self.low_rank, bias=False) for i in range(self.group_num)])
        self.Ws = nn.ModuleList([nn.Linear(self.low_rank, self.group_param_sizes[i], bias=False) for i in range(self.group_num)])

        self.UNK_EMBED = np.random.random(size=(self.word_embed_size,))
        self.SOS_EMBED = np.random.random(size=(self.word_embed_size,))
        self.EOS_EMBED = np.random.random(size=(self.word_embed_size,))
        self.PAD_EMBED = np.random.random(size=(self.word_embed_size,))
        # lang_set = set()
        # for pair in args['--langs'].split(','):
        #     lang1, lang2 = pair.split('-')
        #     lang_set.add(lang1)
        #     lang_set.add(lang2)
        # self.id2lang = list(lang_set)
        # self.lang2id = {lang: i for i, lang in enumerate(self.id2lang)}
        # init language embeddings
        self.word_embeddings = [Embed(lang, vocabs[LANG_INDICES[lang]], self.vocab_size, self.word_embed_size)
                                for lang in LANG_NAMES]
        # initialize the parameters using uniform distribution
        for param in self.parameters():
            nn.init.uniform_(param.data, a=-0.4, b=0.4)

    def create_embed_layer(self, lang, non_trainable=False):
        weights_path = 'embed/%s.embed.npy' % lang
        if os.path.isfile(weights_path):
            weights_matrix = np.load(weights_path)
        else:
            model = fastText.load_model('embed/%s.embed.bin' % lang)
            model.get_words()
        emb_layer = nn.Embedding(self.vocab_size, self.word_embed_size)
        emb_layer.weight = nn.Parameter(torch.from_numpy(weights_matrix).float())
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer

    @staticmethod
    def get_param_meta(shapes: List[List[Tuple[int]]]):
        # calculate the parameters groups sizes and numbers
        group_num = len(shapes)
        # a list of param number in each group [[1024, 1024], [5120, 2560, 2560] ...]umber for each group
        group_param_num = []
        # a list of TOTAL param number in each group [2048, 10240, ...]
        group_param_sizes = []

        for group in shapes:
            group_param = []
            group_param_size = 0
            for shape in group:
                shape_size = reduce(lambda product, dim: product * dim, shape, 1)
                group_param.append(shape_size)
                group_param_size += shape_size

            group_param_num.append(group_param)
            group_param_sizes.append(group_param_size)

        return group_num, group_param_num, group_param_sizes

    def get_params(self, langs: List[int]) -> List[List[Tensor]]:
        """
        Gets the grouped parameters required by the model

        Args:
            langs: a list of language indices representing the language using utils.LANG_INDICES

        Return:
            grouped_params: a list of groups of parameters in tensor form
        """
        assert (len(langs) == self.group_num)

        # generate parameters for this language by group
        params = []
        for j in range(self.group_num):
            ell_j = self.L(self.lang_encode[langs[j]])
            P_j = self.Ps[j]
            W_j = self.Ws[j]
            W_j_P_j_ell_j = W_j(P_j(ell_j))
            params.append(W_j_P_j_ell_j)

        # separate the params inside the group and reshape to desired shape
        for j in range(self.group_num):
            vecs_in_group = torch.split(params[j], self.group_param_num[j], dim=0)

            tensors_in_group = []
            for i in range(len(vecs_in_group)):
                tsr = vecs_in_group[i].reshape(self.shapes[j][i])
                tensors_in_group.append(tsr)
            params[j] = tensors_in_group
        return params

    def get_embedding(self, lang: int):
        # get the word embedding for the language
        word_embedding = self.word_embeddings[lang]

        return word_embedding

    @DeprecationWarning
    def forward(self, L, X, y):
        pass


if __name__ == '__main__':
    args = dict()
    args['--lang_embed_size'] = 8
    args['--embed-size'] = 256
    args['--vocab_size'] = 20000
    args['--low_rank'] = 4

    shapes = [[(10, 20), (10, 20, 30), (10, 20, 30)], [(100, 200), (10, 2, 300)], [(1, 20, 45), (10, 2)]]

    print(CPG.get_param_meta(shapes))

    cpg = CPG(shapes, args)
    result = cpg.get_params([1, 1, 0])

    for tensor_list in result:
        print('%d tensors in this group' % len(tensor_list))
        for tsr in tensor_list:
            print(tsr.shape)

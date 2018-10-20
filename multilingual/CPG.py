from typing import *

import torch
import torch.nn as nn
import torch.tensor as Tensor
from functools import reduce

LANG_INDICES = {'en': 0,
                'gl': 1, 'pt': 4,
                'az': 2, 'tr': 5,
                'be': 3, 'ru': 6}


class CPG(nn.Module):
    def __init__(self, shapes: List[List[Tuple[int]]], size_dict: Dict[str, int]):
        """
        Args:
            shapes: List[List[tuples]] a list of groups, where each tuple the
                    shape of the params in that group
        """
        super(CPG, self).__init__()

        # init size constants

        self.lang_embed_size = size_dict['lang_embed_size']
        self.word_embed_size = size_dict['word_embed_size']
        self.num_lang = size_dict['num_lang']
        self.vocab_size = size_dict['vocab_size']
        self.low_rank = size_dict['low_rank']
        self.lang_encode = torch.eyes(self.num_lang)

        self.shapes = shapes
        self.group_num, self.group_param_num, self.group_param_sizes = self.get_param_meta(shapes)
        
        # init every layer of CPG for different groups
        self.L = nn.Linear(self.num_lang, self.lang_embed_size, bias=False)
        self.Ps = nn.ModuleList([nn.Linear(self.lang_embed_size, self.low_rank) for _ in range(self.group_num)])
        self.Ws = nn.ModuleList([nn.Linear(self.low_rank, self.group_param_sizes[i]) for i in range(self.group_num)])

        # init language embeddings
        self.word_embeddings = nn.ModuleList([nn.Embedding(self.vocab_size, self.word_embed_size)
                                              for _ in range(self.num_lang)])

        # initialize the parameters using uniform distribution
        for param in self.parameters():
            nn.init.uniform_(param.data, a=-0.1, b=0.1)

    @staticmethod
    def get_param_meta(shapes: List[List[Tuple[int]]]):
        # calculate the parameters groups sizes and numbers
        group_num = len(shapes)
        # a list of param number in each group [[1024, 1024], [5120, 2560, 2560] ...]umber for each group
        group_param_num = []
        # a list of TOTAL param number in each group [2048, 10240, ...]
        group_param_sizes = []

        for group in shapes:
            group_param_num.append(len(group))
            group_param_size = 0
            for shape in group:
                group_param_size += reduce(lambda product, dim: product * dim, shape, 1)

            group_param_sizes.append(group_param_size)

        return group_num, group_param_num, group_param_sizes

    def get_params(self, langs: List[int]) -> List[List[Tensor]]:
        """
        Gets the grouped parameters required by the model

        Args:
            langs: a list of language indices representing the language using CPG.LANG_INDICES

        Return:
            grouped_params: a list of groups of parameters in tensor form
        """
        assert(len(langs) == self.group_num)

        # generate parameters for this language by group
        params = []
        for j in range(self.group_num):
            ell_j = self.L(self.lang_encode[langs[j]])
            P_j = self.Ps[j]
            W_j = self.Ws[j]
            P_j_ell_j = P_j(ell_j)
            W_j_P_j_ell_j = W_j(P_j_ell_j)
            params.append(W_j_P_j_ell_j)

        # separate the params inside the group and reshape to desired shape
        grouped_params = []
        for j in range(self.group_num):
            vecs_in_group = torch.split(params[j], self.group_param_num[j], dim=0)

            tensors_in_group = []
            for i in range(len(vecs_in_group)):
                tsr = vecs_in_group[i].reshape(self.shapes[j][i])
                tensors_in_group.append(tsr)
            grouped_params.append(tensors_in_group)
        return grouped_params

    def get_embedding(self, lang: int):
        # get the word embedding for the language
        word_embedding = self.word_embeddings[lang]

        return word_embedding

    @DeprecationWarning
    def forward(self, L, X, y):
        pass 

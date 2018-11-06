import torch
from collections import namedtuple, OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

LANG_NAMES = ['en', 'az', 'be', 'gl', 'tr',  'ru',  'pt']

LANG_INDICES = {l: i for i, l in enumerate(LANG_NAMES)}

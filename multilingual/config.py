import torch
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

LANG_NAMES = {0: 'en',
              1: 'az',
              2: 'tr'}

LANG_INDICES = {v: k for k, v in LANG_NAMES.items()}

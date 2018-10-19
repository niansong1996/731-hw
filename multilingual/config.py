import torch
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

LANG_INDICES = { 0 : 'en', 
                 1 : 'gl', 4 : 'pt',
                 2 : 'az', 5 : 'tr',
                 3 : 'be', 6 : 'ru' }
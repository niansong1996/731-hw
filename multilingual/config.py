import torch
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

LANG_INDICES = { 0 : 'en', 
                 1 : 'az', 4 : 'tr',
                 2 : 'be', 5 : 'ru', 
                 3 : 'gl', 6 : 'pt' }

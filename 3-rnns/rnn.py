# coding: utf-8

import torch
from torch import nn
import glob
import unicodedata
import string
import numpy as np
from torch.autograd import Variable
import itertools
import sys

def findFiles(path): return glob.glob(path)

all_letters = string.ascii_letters + ',.; \''

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

language_to_names = {}
for file in findFiles('data/names/*.txt'):
    with open(file) as infile:
        language = file.split('/')[-1].replace('.txt', '')
        language_to_names[language] = [unicodeToAscii(x.strip()) for x in infile.readlines()]

HIDDEN_DIM = 128
CHAR_DIM = len(all_letters)
LANGUAGE_DIM = len(language_to_names)
COMBINATION_DIM = CHAR_DIM + HIDDEN_DIM

_tensor_index_by_char = {}
_char_by_tensor_index = {}
for i, c in enumerate(all_letters):
    _tensor_index_by_char[c] = i
    _char_by_tensor_index[i] = c
_char_tensors_by_index = np.identity(CHAR_DIM)

def char_to_tensor(c):
    return _char_tensors_by_index[_tensor_index_by_char[c]]

def name_to_tensor(name):
    return torch.Tensor([char_to_tensor(c) for c in name])

_language_tensors_by_index = np.identity(LANGUAGE_DIM)

language_list = list(language_to_names.keys())

_tensor_index_by_language = {}
_language_by_tensor_index = {}
for i, language in enumerate(language_list):
    _tensor_index_by_language[language] = i
    _language_by_tensor_index[i] = language

def language_to_tensor(language):
    return torch.Tensor(_language_tensors_by_index[_tensor_index_by_language[language]])

def tensor_to_language(x):
    return asdflkjdsfdas


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_output = nn.Linear(COMBINATION_DIM, LANGUAGE_DIM)
        self.input_to_hidden = nn.Linear(COMBINATION_DIM, HIDDEN_DIM)
        self.log_softmax = nn.LogSoftmax()
        self.zero()

    def forward(self, input, hidden):
        combination = torch.cat((input, hidden))
        output = self.input_to_output(combination.t())
        new_hidden = self.input_to_hidden(combination.t()).t()
        transformed_output = self.log_softmax(output)
        return transformed_output, new_hidden

    def zero(self):
        self._zero_grad()
        self._zero_hidden()

    def _zero_grad(self):
        self.input_to_output.zero_grad()
        self.input_to_hidden.zero_grad()
        self.log_softmax.zero_grad()

    def _zero_hidden(self):
        self.hidden = Variable(torch.zeros(HIDDEN_DIM, 1))

def flatten(iterable):
    return list(itertools.chain(*iterable))

_train_pairs = flatten([list(zip(len(names) * [language], names) )for language, names in language_to_names.items()])
all_training_examples = [(_tensor_index_by_language[x], name_to_tensor(y)) for (x, y) in _train_pairs]

def train(num_iters=100000, print_iters=5000, learning_rate=0.01):
    rnn = RNN()
    loss_function = nn.NLLLoss()
    for i in range(num_iters):
        for label, name in all_training_examples:
            rnn.zero()
            for c in name:
                out, hidden = rnn(Variable(c.unsqueeze(1)), hidden)
            loss_result = loss_function(out, Variable(torch.LongTensor([label])))
            loss_result.backward(retain_graph=True)
            for param in rnn.parameters():
                param.data -= param.grad.data * learning_rate
            loss = loss_result.data[0]
            print("{}: {}".format(i, loss))
            if loss < 1e-6:
                sys.exit(0)

train()

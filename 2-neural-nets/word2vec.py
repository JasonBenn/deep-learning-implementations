import torch
from torch import LongTensor, FloatTensor
from torch.autograd import Variable
from itertools import islice

words = ["a", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
one_hot_encodings = torch.eye(len(words))
word_to_one_hot = {}
for i, word in enumerate(words):
    word_to_one_hot[word] = one_hot_encodings[i]

words_to_indexes = { word: i for i, word in enumerate(words) }


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


class Word2Vec(torch.nn.Module):
    def __init__(self, embedding_size, vocab_size, cbow_window_size=4):
        super(Word2Vec, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.layer1 = torch.nn.Linear(embedding_size * cbow_window_size, embedding_size)
        self.layer2 = torch.nn.Linear(embedding_size, vocab_size)

    def forward(self, inputs):
        input_embeddings = self.embedding(inputs).view((1, -1))  # Reshape to 36,1
        activations_1 = torch.sigmoid(self.layer1(input_embeddings))
        return torch.nn.Softmax()(self.layer2(activations_1))

EMBEDDING_SIZE = 200
model = Word2Vec(EMBEDDING_SIZE, len(words))

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=.001)

for epoch in range(5):
    word_iter = window(words, 5)
    total_loss = 0
    for i in range(5):
        optimizer.zero_grad()
        example = next(word_iter)
        word_indexes = [words_to_indexes[word] for word in example]
        inputs = Variable(torch.cat((LongTensor(word_indexes[:2]), LongTensor(word_indexes[3:]))))
        label = Variable(word_to_one_hot[example[2]])
        predictions = model(inputs)
        loss = loss_fn(predictions, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    print("epoch {} loss: {}".format(epoch, total_loss))

print("✅")

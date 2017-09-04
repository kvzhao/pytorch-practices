import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

x = Variable(torch.randn(1, 10))

rnn = nn.LSTM(10, 20)

out, h = rnn(x)

print (out)

num_embedding = 2
# dim of embedded output layer
embeded_dim = 10

ising_like_state = Variable(torch.LongTensor([[1,0,1,1,0,1,0,1,0,1]]))

embed = nn.Embedding(num_embedding, embeded_dim)
print (ising_like_state)

e = embed(ising_like_state)
print (e)

out, h = rnn(e)
print (out)
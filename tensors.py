from __future__ import print_function
import torch 

x = torch.randn(3, 5)
print (x)
print (x.size())

x.add_(1)
print(x)

# no camel casing

a = torch.ones(5, 5)
b = a.numpy()

b[1, 1] = 2
print (b)
print (a)
# no memory copy, by reference
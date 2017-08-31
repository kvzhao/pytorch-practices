from __future__ import print_function
import torch 

x = torch.randn(3, 5)
print (x)
print (x.size())

x.add_(1)
print(x)

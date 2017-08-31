import torch 
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
print (x)
print (x.grad_fn)

y =  x + 2
print (y)
print (y.grad_fn)

z = y ** 2 * 3
print (z)
print (z.grad_fn)
out = z.mean()

print(out)
print (out.grad_fn)

print (x.grad)
out.backward()
print (x.grad)
print (y.grad)
print (z.grad)
print (out.grad)

#print (x.view(x.size()))
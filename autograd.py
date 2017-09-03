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

x = Variable(torch.randn(1, 10))
prev_h = Variable(torch.randn(1, 20))
W_h = Variable(torch.randn(20, 20))
W_x = Variable(torch.randn(20, 10))

# input to hidden
i2h = torch.mm(W_x, x.t())
h2h = torch.mm(W_h, prev_h.t())
next_h = i2h + h2h
next_h = next_h.tanh()

next_h.backward(torch.ones(1, 20))

print (next_h)
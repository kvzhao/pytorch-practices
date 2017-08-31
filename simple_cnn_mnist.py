import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        print (self.conv1)
        print (type(self.conv1))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input)))
        x = self.pool2(F.relu(self.conv2(x)))
        # tensor.view(): Returns a new tensor with the same data but different size.
        x = x.view(x.size(0), -1)
        # get fc1 layer features
        y = F.relu(self.fc1(x))
        x = F.relu(self.fc2(y))
        return x, y

net = MNISTConvNet()
print(net)

input = Variable(torch.randn(1, 1, 28, 28))
out, fc1 = net(input)
print (out.size())
print (out)

#print (fc1)
print (fc1.size())

target = Variable(torch.LongTensor([3]))
loss_fn = nn.CrossEntropyLoss() # LogSoftmax + ClassNLL Loss
err = loss_fn(out, target)
err.backward()
print(err)

# get intermediate layers' information
print(net.conv1.weight.grad.size())
print(net.conv1.weight.data.norm())  # norm of the weight
print(net.conv1.weight.grad.data.norm())  # norm of the gradients

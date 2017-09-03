import torch
import torch.nn as nn
from torch.autograd import Variable

# Hyper Parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = Net(input_size, hidden_size, num_classes)
print (net)
print (net.fc1)
print (net.fc2)


dummy_batch = 5
dummy_x = Variable(torch.randn(dummy_batch, input_size))

print (net(dummy_x))

err_measure = nn.CrossEntropyLoss()
print (net.parameters())
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
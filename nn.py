#http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
t1 = net.parameters()
t2 = list(t1)
params = list(net.parameters())
print(len(params))
for idx in range(len(params)):
    print(params[idx].size())
input = Variable(torch.randn(1, 1, 32, 32))
output = net(input)
print(output)

target = Variable(torch.arange(1, 11))
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

print(loss.creator)  # MSELoss
print(loss.creator.previous_functions[0][0])  # Linear
print(loss.creator.previous_functions[0][0].previous_functions[0][0])  # ReLU

net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)

optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
print('conv1.bias before backward')
print(net.conv1.bias)
print('conv1.bias.data before backward')
print(net.conv1.bias.data)
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
t5 = loss.backward()
print('conv1.bias.grad after backward')
t8 = net.conv1.bias.grad.data.clone()
print(net.conv1.bias.grad)

print('conv1.bias afger backward')
t7 = net.conv1.bias.data.clone()
print(net.conv1.bias)
t6 = optimizer.step()
print('conv1.bias after otim.')
t9 = t7 - t8 * lr
print(net.conv1.bias)
print(t9)
t10 = t9 == net.conv1.bias.data
t11 = t10.prod()
print ('Is the "new bias" equal to "old bias - learning rate * gradient : {}'.format(1 == t11))

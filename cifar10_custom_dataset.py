# tutorial web site : http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

from PIL import Image
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils_data

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from lr_scheduler import ReduceLROnPlateau

# functions to show an image

class CustomDataSet(utils_data.Dataset):
    def __init__(self, config):
        self.dataset_path = config['dataset_path']
        self.num_samples = config['train_data_size']
        self.ids_list = list(range(1, self.num_samples + 1))
        random.shuffle(self.ids_list)

    def __getitem__(self, index):
        image = Image.open('{}train/{:>06}.png'.format(self.dataset_path, self.ids_list[index]))
        image = np.array(image)
        image = np.rollaxis(image, 2, 0)
        label = np.load('{}train_label(pytorch)/{:>06}.npy'.format(self.dataset_path, self.ids_list[index]))
        image = np.array(image).astype(np.float32)
        label = np.array(label).astype(np.int)
        return image, label

    def __len__(self):
        return len(self.ids_list)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        t1 = self.num_flat_features(x)
        x = x.view(-1, self.num_flat_features(x))
        #x = x.view(-1, 16 * 5 * 5)
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


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #plt.imshow(npimg)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

n_epoch = 100

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# the size of CIFAR10 dataset : around 341 MB
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                        download=True, transform=transform)

config = {'dataset_path':'./data', 'train_data_size':100}
trainset = CustomDataSet(config)

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                          shuffle=True, num_workers=2)

trainloader = utils_data.DataLoader(trainset, batch_size=4,
    shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#'''

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#'''

#net = Net().cuda()
net = Net()
#t1 = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=1) # set up scheduler

n_image_total = 0
running_loss = 0.0
is_lr_just_decayed = False
shall_stop = False
for epoch in range(n_epoch):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        #inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        #labels += 10
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #n_image_total += labels.size()[0]
        # print statistics
        running_loss += loss.data[0]
        if n_image_total % 2000 == 1999:    # print every 2000 mini-batches
        #if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            is_best_changed, is_lr_decayed = scheduler.step(running_loss / 2000, n_image_total + 1) # update lr if needed
            if is_lr_just_decayed and (not is_best_changed):
                shall_stop = True
                break
            is_lr_just_decayed = is_lr_decayed
            running_loss = 0.0

        n_image_total += 1
    if shall_stop:
        break

print('Finished Training')

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


outputs = net(Variable(images.cuda()))
_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j][0]]
                              for j in range(4)))


correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

for data in testloader:
    images, labels = data
    #images, labels = images.cuda(), labels.cuda()
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
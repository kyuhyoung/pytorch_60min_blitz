# tutorial web site : http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# custom dataset site : http://kevin-ho.website/Make-a-Acquaintance-with-Pytorch/

import os
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
from install_cifar10 import prepare_cifar10_dataset


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #plt.imshow(npimg)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# functions to show an image
def get_exact_file_name_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]

def make_dataloader_torchvison_memory(dir_data, transform):
    # the size of CIFAR10 dataset : around 341 MB
    trainset = torchvision.datasets.CIFAR10(root=dir_data, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=dir_data, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

def make_dataloader_torchvison_imagefolder(dir_data, transform):

    #execfile('install_cifar10.py')
    prepare_cifar10_dataset(dir_data)

    trainset = torchvision.datasets.CIFAR10(root=dir_data, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=dir_data, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

def make_dataloader_custom_memory():

    execfile('install_cifar10.py')

    config_train = {'dataset_path': './data/train', 'fn_label': './data/train_map.txt',
                    'data_size': 50000}
    config_test = {'dataset_path': './data/test', 'fn_label': './data/test_map.txt',
                   'data_size': 10000}
    # config = {'dataset_path':'./data/train'}
    trainset = Cifar10CustomMemory(config_train)
    trainloader = utils_data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=2)

    testset = Cifar10CustomMemory(config_test)
    testloader = utils_data.DataLoader(testset, batch_size=4,
                                       shuffle=False, num_workers=2)
    return trainloader, testloader

def make_dataloader_custom_file():

    execfile('install_cifar10.py')

    config_train = {'dataset_path': './data/train', 'fn_label': './data/train_map.txt',
                    'data_size': 50000}
    config_test = {'dataset_path': './data/test', 'fn_label': './data/test_map.txt',
                   'data_size': 10000}
    # config = {'dataset_path':'./data/train'}
    trainset = Cifar10CustomFile(config_train)
    trainloader = utils_data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=2)

    testset = Cifar10CustomFile(config_test)
    testloader = utils_data.DataLoader(testset, batch_size=4,
                                       shuffle=False, num_workers=2)
    return trainloader, testloader


class Cifar10CustomMemory(utils_data.Dataset):
    def __init__(self, config):
        self.dataset_path = config['dataset_path']
        self.num_samples = config['data_size']
        self.fn_label = config['fn_label']
        #self.ids_list = list(range(1, self.num_samples + 1))
        self.ids_list = list(range(self.num_samples))
        li_tokens = [line.strip().split("\t") for line in open(self.fn_label)]
        self.di_id_label = {get_exact_file_name_from_path(tokens[0]):int(tokens[1]) for tokens in li_tokens}
        #random.shuffle(self.ids_list)
        return

    def __getitem__(self, index):
        id_img = self.ids_list[index]
        str_id = '{:>05}'.format(id_img)
        fn_img = '{}/{}.png'.format(self.dataset_path, str_id)
        image = Image.open(fn_img)
        image = np.array(image)
        image = np.rollaxis(image, 2, 0)
        image = image / 255.
        #fn_label = '{}train_label(pytorch)/{:>05}.npy'.format(self.dataset_path, self.ids_list[index])
        label = self.di_id_label[str_id]
        #print('Read image of label %d as input : %s' % (label, fn_img))
        #'''
        image = np.array(image).astype(np.float32)
        #label = np.array(label).astype(np.int)
        #'''
        return image, label

    def __len__(self):
        return len(self.ids_list)


class Cifar10CustomFile(utils_data.Dataset):
    def __init__(self, config):
        self.dataset_path = config['dataset_path']
        self.num_samples = config['data_size']
        self.fn_label = config['fn_label']
        #self.ids_list = list(range(1, self.num_samples + 1))
        self.ids_list = list(range(self.num_samples))
        li_tokens = [line.strip().split("\t") for line in open(self.fn_label)]
        self.di_id_label = {get_exact_file_name_from_path(tokens[0]):int(tokens[1]) for tokens in li_tokens}
        #random.shuffle(self.ids_list)
        return

    def __getitem__(self, index):
        id_img = self.ids_list[index]
        str_id = '{:>05}'.format(id_img)
        fn_img = '{}/{}.png'.format(self.dataset_path, str_id)
        image = Image.open(fn_img)
        image = np.array(image)
        image = np.rollaxis(image, 2, 0)
        image = image / 255.
        #fn_label = '{}train_label(pytorch)/{:>05}.npy'.format(self.dataset_path, self.ids_list[index])
        label = self.di_id_label[str_id]
        #print('Read image of label %d as input : %s' % (label, fn_img))
        #'''
        image = np.array(image).astype(np.float32)
        #label = np.array(label).astype(np.int)
        #'''
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



def main():

    TORCHVISION_MEMORY = 1
    TORCHVISION_IMAGEFOLDER = 2
    CUSTOM_MEMORY = 3
    CUSTOM_FILE = 4

    #mode = TORCHVISION_MEMORY
    mode = TORCHVISION_IMAGEFOLDER
    #mode = CUSTOM_MEMORY
    #mode = CUSTOM_FILE

    dir_data = './data'
    n_epoch = 100

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if TORCHVISION_MEMORY == mode:
        trainloader, testloader = make_dataloader_torchvison_memory(dir_data, transform)
    elif TORCHVISION_IMAGEFOLDER == mode:
        trainloader, testloader = make_dataloader_torchvison_imagefolder(dir_data, transform)
    elif CUSTOM_MEMORY == mode:
        trainloader, testloader = make_dataloader_custom_memory()
    else:
        trainloader, testloader = make_dataloader_custom_file()



    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    #'''

    # get some random training images
    #dataiter = iter(trainloader)
    #images, labels = dataiter.next()

    # show images
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    #print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
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
        print('epoch : %d' % (epoch + 1))
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

    '''
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


    outputs = net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j][0]]
                                  for j in range(4)))
    '''

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
    return

if __name__ == "__main__":
    main()

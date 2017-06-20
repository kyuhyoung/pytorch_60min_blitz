# tutorial web site : http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# custom dataset site : http://kevin-ho.website/Make-a-Acquaintance-with-Pytorch/

import os
from os.path import join
from os import makedirs, listdir
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
from torchvision import datasets

import matplotlib.pyplot as plt
import numpy as np
from lr_scheduler import ReduceLROnPlateau
from install_cifar10 import prepare_cifar10_dataset
from time import time


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #plt.imshow(npimg)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# functions to show an image
def get_exact_file_name_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]

def make_dataloader_torchvison_memory(dir_data, di_set_transform):
    # the size of CIFAR10 dataset : around 341 MB
    trainset = torchvision.datasets.CIFAR10(
        root=dir_data, train=True, download=True, transform=di_set_transform['train'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(
        root=dir_data, train=False, download=True, transform=di_set_transform['test'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    li_class = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, li_class

def make_dataloader_torchvison_imagefolder(dir_data, data_transforms, ext_img):

    li_class = prepare_cifar10_dataset(dir_data, ext_img)
    li_set = ['train', 'test']
    dsets = {x: datasets.ImageFolder(join(dir_data, x), data_transforms[x])
             for x in li_set}
    dset_loaders = {x: torch.utils.data.DataLoader(
        dsets[x], batch_size=4, shuffle=True, num_workers=4) for x in li_set}

    trainloader, testloader = dset_loaders[li_set[0]], dset_loaders[li_set[1]]
    return trainloader, testloader, li_class

def make_dataloader_custom_memory(dir_data, data_transforms, ext_img):

    li_class = prepare_cifar10_dataset(dir_data, ext_img)
    li_set = ['train', 'test']
    data_size = {'train' : 50000, 'test' : 10000}
    dsets = {x: Cifar10CustomMemory(
        join(dir_data, x), data_size[x], data_transforms[x], li_class, ext_img)
             for x in li_set}
    dset_loaders = {x: torch.utils.data.DataLoader(
        dsets[x], batch_size=4, shuffle=True, num_workers=4) for x in li_set}
    trainloader, testloader = dset_loaders[li_set[0]], dset_loaders[li_set[1]]

    return trainloader, testloader, li_class

def make_dataloader_custom_file(dir_data, data_transforms, ext_img):

    li_class = prepare_cifar10_dataset(dir_data, ext_img)
    li_set = ['train', 'test']
    data_size = {'train' : 50000, 'test' : 10000}
    dsets = {x: Cifar10CustomFile(
        join(dir_data, x), data_size[x], data_transforms[x], li_class, ext_img)
             for x in li_set}
    dset_loaders = {x: utils_data.DataLoader(
        dsets[x], batch_size=4, shuffle=True, num_workers=4) for x in li_set}
    trainloader, testloader = dset_loaders[li_set[0]], dset_loaders[li_set[1]]

    return trainloader, testloader, li_class

def make_dataloader_custom_tensordataset(dir_data, data_transforms, ext_img):

    li_class = prepare_cifar10_dataset(dir_data, ext_img)
    n_class = len(li_class)
    li_set = ['train', 'test']
    '''
    features = {}
    for set in li_set:
        features[set] = [data_transforms[x](Image.open(join(join(join(dir_data, x), label), fn_img)).convert('RGB')) for fn_img in
                            listdir(join(join(dir_data, x), label))
                            if fn_img.endswith(ext_img) for label in li_class]
    '''
    features = {x : [data_transforms[x](Image.open(join(join(join(dir_data, x), label), fn_img)).convert('RGB'))
                     for label in li_class for fn_img in listdir(join(join(dir_data, x), label)) if fn_img.endswith(ext_img)] for x in li_set}
    a = 0
    b = 0
    targets = {x : [i_l for fn_img in
                            listdir(join(join(dir_data, x), label))
                            if fn_img.endswith(ext_img)] for x in li_set for i_l, label in enumerate(li_class)}
    dsets = {x: utils_data.TensorDataset(features[x], targets[x]) for x in li_set}
    dset_loaders = {x: utils_data.DataLoader(
        dsets[x], batch_size=4, shuffle=True, num_workers=4) for x in li_set}
    trainloader, testloader = dset_loaders[li_set[0]], dset_loaders[li_set[1]]
    return trainloader, testloader, li_class


class Cifar10CustomMemory(utils_data.Dataset):
    def __init__(self, dataset_path, data_size, data_transform, li_label, ext_img):

        self.dataset_path = dataset_path
        self.num_samples = data_size
        self.transform = data_transform
        #for label in li_label:
        self.li_img_classid = []
        for idx, label in enumerate(li_label):
            print('dumping images of [%s] into memory' % (label))
            dir_label = join(dataset_path, label)
            #li_fn_img = [file for file in listdir(dir_label) if file.endswith(ext_img)]
            self.li_img_classid += [(Image.open(join(dir_label, fn_img)).convert('RGB'), idx) for fn_img in listdir(dir_label)
                      if fn_img.endswith(ext_img)]
        return

    def __getitem__(self, index):

        img, target = self.li_img_classid[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        #return len(self.ids_list)
        return len(self.li_img_classid)


class Cifar10CustomFile(utils_data.Dataset):
    def __init__(self, dataset_path, data_size, data_transform, li_label, ext_img):
        self.dataset_path = dataset_path
        self.num_samples = data_size
        self.transform = data_transform
        self.li_fn_img_classid = []
        for idx, label in enumerate(li_label):
            dir_label = join(dataset_path, label)
            self.li_fn_img_classid += [(join(dir_label, fn_img), idx) for fn_img in listdir(dir_label)
                                       if fn_img.endswith(ext_img)]
        return

    def __getitem__(self, index):

        fn_img, target = self.li_fn_img_classid[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)
        img = Image.open(fn_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.li_fn_img_classid)


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


def initialize(mode, dir_data, di_set_transform, ext_img):

    if 'TORCHVISION_MEMORY' == mode:
        trainloader, testloader, li_class = make_dataloader_torchvison_memory(
            dir_data, di_set_transform)
    elif 'TORCHVISION_IMAGEFOLDER' == mode:
        trainloader, testloader, li_class = make_dataloader_torchvison_imagefolder(
            dir_data, di_set_transform, ext_img)
    elif 'CUSTOM_MEMORY' == mode:
        trainloader, testloader, li_class = make_dataloader_custom_memory(
            dir_data, di_set_transform, ext_img)
    elif 'CUSTOM_FILE' == mode:
        trainloader, testloader, li_class = make_dataloader_custom_file(
            dir_data, di_set_transform, ext_img)
    else:
        trainloader, testloader, li_class = make_dataloader_custom_tensordataset(
            dir_data, di_set_transform, ext_img)


    #net = Net().cuda()
    net = Net()
    #t1 = net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=1) # set up scheduler

    return trainloader, testloader, net, criterion, optimizer, scheduler, li_class

def train(trainloader, testloader, net, criterion, optimizer, scheduler,
          li_class, n_epoch, lap_init, ax_time, ax_loss, kolor):

    n_image_total = 0
    i_batch = 0
    running_loss = 0.0
    li_i_batch = []
    li_loss_avg = []
    li_lap = [lap_init]
    li_epoch = [0]
    is_lr_just_decayed = False
    shall_stop = False
    ax_time.plot(li_epoch, li_lap, c=kolor)
    plt.pause(0.05)
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        print('epoch : %d' % (epoch + 1))
        start_epoch = time()
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

                running_loss_avg = running_loss / 2000
                li_i_batch.append(i_batch)
                li_loss_avg.append(running_loss_avg)
                ax_loss.plot(li_i_batch, li_loss_avg, c=kolor)
                plt.pause(0.05)
                i_batch += 1

                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss_avg))
                is_best_changed, is_lr_decayed = scheduler.step(
                    running_loss_avg, n_image_total + 1) # update lr if needed
                if i_batch >= 3:
                    break
                if is_lr_just_decayed and (not is_best_changed):
                    shall_stop = True
                    break
                is_lr_just_decayed = is_lr_decayed
                running_loss = 0.0

            n_image_total += 1
        lap_epoch = time() - start_epoch
        lap_batch = lap_epoch / (i + 1)
        li_lap.append(lap_batch)
        li_epoch.append(epoch + 1)
        ax_time.plot(li_epoch, li_lap, c=kolor)
        plt.pause(0.05)
        if shall_stop:
            break

    print('Finished Training')

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

    for i, klass in enumerate(li_class):
        print('Accuracy of %5s : %2d %%' % (
            klass, 100 * class_correct[i] / class_total[i]))
    '''
    return

def main():

    li_mode = ['TORCHVISION_MEMORY', 'TORCHVISION_IMAGEFOLDER',
               'CUSTOM_MEMORY', 'CUSTOM_FILE', 'CUSTOM_TENSORDATSET']
    '''
    #mode = TORCHVISION_MEMORY
    #mode = TORCHVISION_IMAGEFOLDER
    mode = CUSTOM_MEMORY
    #mode = CUSTOM_FILE
    '''
    dir_data = './data'
    ext_img = 'png'
    #n_epoch = 100
    n_epoch = 1

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    di_set_transform = {'train' : transform, 'test' : transform}

    fig = plt.figure()#num=None, figsize=(20, 30), dpi=500)
    plt.ion()
    ax_time = fig.add_subplot(2, 1, 1)
    ax_loss = fig.add_subplot(2, 1, 2)
    for i_m, mode in enumerate(li_mode):
        start = time()
        trainloader, testloader, net, criterion, optimizer, scheduler, li_class = \
            initialize(mode, dir_data, di_set_transform, ext_img)
        lap_init = time() - start
        #print('[%s] lap of initializing : %d sec' % (lap_sec))
        kolor = np.random.rand(3)
        if 2 == i_m:
            a = 0
        train(trainloader, testloader, net, criterion, optimizer, scheduler,
              li_class, n_epoch, lap_init, ax_time, ax_loss, kolor)

    return

if __name__ == "__main__":
    main()

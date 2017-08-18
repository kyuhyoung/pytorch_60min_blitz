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
from matplotlib.ticker import MaxNLocator
import collections, sys

if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    #plt.imshow(npimg)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# functions to show an image
def get_exact_file_name_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]

def make_dataloader_torchvison_memory(dir_data, di_set_transform,
                                      n_img_per_batch, n_worker):
    # the size of CIFAR10 dataset : around 341 MB
    trainset = torchvision.datasets.CIFAR10(
        root=dir_data, train=True, download=True, transform=di_set_transform['train'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=n_img_per_batch,
                                              shuffle=True, num_workers=n_worker)
    testset = torchvision.datasets.CIFAR10(
        root=dir_data, train=False, download=True, transform=di_set_transform['test'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=n_img_per_batch,
                                             shuffle=False, num_workers=n_worker)
    li_class = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, li_class

def make_dataloader_torchvison_imagefolder(dir_data, data_transforms, ext_img,
                                           n_img_per_batch, n_worker):

    li_class = prepare_cifar10_dataset(dir_data, ext_img)
    li_set = ['train', 'test']
    dsets = {x: datasets.ImageFolder(join(dir_data, x), data_transforms[x])
             for x in li_set}
    dset_loaders = {x: torch.utils.data.DataLoader(
        dsets[x], batch_size=n_img_per_batch, shuffle=True, num_workers=n_worker) for x in li_set}

    trainloader, testloader = dset_loaders[li_set[0]], dset_loaders[li_set[1]]
    return trainloader, testloader, li_class

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

def my_collate(batch):
    _use_shared_memory = True
    "Puts each data field into a tensor with outer dimension batch size"
    if torch.is_tensor(batch[0]):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        if type(elem).__name__ == 'ndarray':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: my_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        t1 = [my_collate(samples) for samples in transposed]
        return [my_collate(samples) for samples in transposed]


def make_dataloader_custom_file(dir_data, data_transforms, ext_img,
                                n_img_per_batch, n_worker):

    li_class = prepare_cifar10_dataset(dir_data, ext_img)
    li_set = ['train', 'test']
    data_size = {'train' : 50000, 'test' : 10000}
    dsets = {x: Cifar10CustomFile(
        join(dir_data, x), data_size[x], data_transforms[x], li_class, ext_img)
             for x in li_set}
    dset_loaders = {x: utils_data.DataLoader(
        dsets[x], batch_size=n_img_per_batch, shuffle=True, num_workers=n_worker, collate_fn=my_collate) for x in li_set}
    trainloader, testloader = dset_loaders[li_set[0]], dset_loaders[li_set[1]]

    return trainloader, testloader, li_class





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
        #img.re
        #margin_x, margin_y = index % 5, (index ** 2) % 5
        margin_x, margin_y = 3, 3
        size_new = tuple(map(sum, zip(img.size, (-margin_x, -margin_y))))
        #size_new = img.size - (margin_x, margin_y)
        img = img.resize(size_new, Image.BICUBIC)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.li_fn_img_classid)


class Net_gap(nn.Module):
    def __init__(self):
        super(Net_gap, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 10, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        t1 = x.size()[2:]
        print('t1')
        print(t1)
        x = F.avg_pool2d(x, t1)
        x = x.view(-1, self.num_flat_features(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def initialize(is_gpu, dir_data, di_set_transform, ext_img, n_img_per_batch, n_worker):

    trainloader, testloader, li_class = make_dataloader_custom_file(
        dir_data, di_set_transform, ext_img, n_img_per_batch, n_worker)

    #net = Net().cuda()
    net = Net_gap()
    #t1 = net.cuda()
    criterion = nn.CrossEntropyLoss()
    if is_gpu:
        net.cuda()
        criterion.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=1, patience = 8, epsilon=0.00001, min_lr=0.000001) # set up scheduler

    return trainloader, testloader, net, criterion, optimizer, scheduler, li_class


def validate_epoch(net, n_loss_rising, loss_avg_pre, ax,
                   li_n_img_val, li_loss_avg_val,
                   testloader, criterion, th_n_loss_rising,
                   kolor, n_img_train, sec, is_gpu):
    net.eval()
    shall_stop = False
    sum_loss = 0
    n_img_val = 0
    start_val = time()
    for i, data in enumerate(testloader):
        inputs, labels = data
        n_img_4_batch = labels.size()[0]
        if is_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        #images, labels = images.cuda(), labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.data[0]
        n_img_val += n_img_4_batch

    lap_val = time() - start_val
    loss_avg = sum_loss / n_img_val
    if loss_avg_pre <= loss_avg:
        n_loss_rising += 1
        if n_loss_rising >= th_n_loss_rising:
            shall_stop = True
    else:
        n_loss_rising = max(0, n_loss_rising - 1)
    li_n_img_val.append(n_img_train)
    li_loss_avg_val.append(loss_avg)
    ax.plot(li_n_img_val, li_loss_avg_val, c=kolor)
    plt.pause(sec)
    loss_avg_pre = loss_avg
    return shall_stop, net, n_loss_rising, loss_avg_pre, ax, \
           li_n_img_val, li_loss_avg_val, lap_val, n_img_val



def test(net, testloader, li_class, is_gpu):

    net.eval()
    n_class = len(li_class)
    correct = 0
    total = 0
    class_correct = list(0. for i in range(n_class))
    class_total = list(0. for i in range(n_class))

    for data in testloader:
        images, labels = data
        if is_gpu:
            images, labels = images.cuda(), labels.cuda()
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







def train_epoch(
        net, trainloader, optimizer, criterion, scheduler, n_img_total,
        n_img_interval, n_img_milestone, running_loss, is_lr_just_decayed,
        li_n_img, li_loss_avg_train, ax_loss_train, sec, epoch,
        kolor, interval_train_loss, is_gpu):
    shall_stop = False
    net.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        n_img_4_batch = labels.size()[0]
        # wrap them in Variable
        # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        if is_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        # labels += 10
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # n_image_total += labels.size()[0]
        # print statistics
        running_loss += loss.data[0]
        #n_image_total += n_img_per_batch
        n_img_total += n_img_4_batch
        n_img_interval += n_img_4_batch

        #if n_image_total % interval_train_loss == interval_train_loss - 1:  # print every 2000 mini-batches
        #if n_image_total % interval_train_loss == 0:  # print every 2000 mini-batches
        if n_img_total > n_img_milestone:  # print every 2000 mini-batches

            # if i % 2000 == 1999:    # print every 2000 mini-batches
            running_loss_avg = running_loss / n_img_interval
            li_n_img.append(n_img_total)
            li_loss_avg_train.append(running_loss_avg)
            ax_loss_train.plot(li_n_img, li_loss_avg_train, c=kolor)
            plt.pause(sec)
            #i_batch += 1
            print('[%d, %5d] avg. loss per image : %.5f' %
                  (epoch + 1, i + 1, running_loss_avg))
            is_best_changed, is_lr_decayed = scheduler.step(
                running_loss_avg, n_img_total)  # update lr if needed
            running_loss = 0.0
            n_img_interval = 0
            n_img_milestone = n_img_total + interval_train_loss
            #'''
            #if is_lr_just_decayed and (not is_best_changed):
            if is_lr_just_decayed and is_lr_decayed:
                shall_stop = True
                break
            #'''
            is_lr_just_decayed = is_lr_decayed
    return shall_stop, net, optimizer, scheduler, n_img_total, n_img_interval, \
           n_img_milestone, running_loss, li_n_img, li_loss_avg_train, ax_loss_train, \
           is_lr_just_decayed, i + 1










def train(is_gpu, trainloader, testloader, net, criterion, optimizer, scheduler, #li_class,
          n_epoch, ax_loss_train, ax_loss_val,
          kolor, interval_train_loss):

    sec = 0.01
    n_image_total = 0
    n_img_interval = 0
    n_img_milestone = interval_train_loss
    running_loss = 0.0
    li_n_img_train, li_n_img_val = [], []
    li_loss_avg_train = []
    li_loss_avg_val = []
    is_lr_just_decayed = False
    #shall_stop = False
    #li_i_epoch = []
    n_loss_rising, th_n_loss_rising, loss_avg_pre = 0, 3, 100000000000
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        print('epoch : %d' % (epoch + 1))
        shall_stop_train, net, optimizer, scheduler, n_image_total, n_img_interval, \
        n_img_milestone, running_loss, li_n_img_train, li_loss_avg_train, ax_loss_train, \
        is_lr_just_decayed, n_batch = train_epoch(
            net, trainloader, optimizer, criterion, scheduler, n_image_total,
            n_img_interval, n_img_milestone, running_loss, is_lr_just_decayed,
            li_n_img_train, li_loss_avg_train, ax_loss_train, sec, epoch,
            kolor, interval_train_loss, is_gpu)
        shall_stop_val, net, n_loss_rising, loss_avg_pre, ax_loss_val, \
        li_n_img_val, li_loss_avg_val, lap_val, n_img_val = \
            validate_epoch(
                net, n_loss_rising, loss_avg_pre, ax_loss_val,
                li_n_img_val, li_loss_avg_val,
                testloader, criterion, th_n_loss_rising, kolor, n_image_total, sec,
                is_gpu)
        #lap_train = time() - start_train
        if shall_stop_train or shall_stop_val:
            break
    ax_loss_train.plot(li_n_img_train, li_loss_avg_train, c=kolor)
    ax_loss_train.legend()
    ax_loss_val.plot(li_n_img_val, li_loss_avg_val, c=kolor)
    ax_loss_val.legend()
    plt.pause(sec)
    print('Finished Training')

    return

def main():

    dir_data = './data'
    ext_img = 'png'
    #n_epoch = 100
    n_epoch = 50
    #n_img_per_batch = 40
    n_img_per_batch = 60
    #n_img_per_batch = 1
    n_worker = 4
    interval_train_loss = int(round(20000 / n_img_per_batch)) * n_img_per_batch
    is_gpu = torch.cuda.device_count() > 0

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    di_set_transform = {'train' : transform, 'test' : transform}

    #fig = plt.figure(num=None, figsize=(1, 2), dpi=500)
    fig = plt.figure(num=None, figsize=(12, 18), dpi=100)
    plt.ion()
    ax_loss_train = fig.add_subplot(2, 1, 1)
    ax_loss_train.set_title('Avg. train loss per image vs. # train input images')
    ax_loss_train.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_loss_val = fig.add_subplot(2, 1, 2)
    ax_loss_val.set_title('Avg. val. loss per image vs. # train input images')
    ax_loss_val.xaxis.set_major_locator(MaxNLocator(integer=True))
    trainloader, testloader, net, criterion, optimizer, scheduler, li_class = \
        initialize(
            is_gpu, dir_data, di_set_transform, ext_img, n_img_per_batch, n_worker)
    #print('[%s] lap of initializing : %d sec' % (lap_sec))
    kolor = np.random.rand(3)
    #if 2 == i_m:
    #    a = 0
    train(is_gpu, trainloader, testloader, net, criterion, optimizer, scheduler, #li_class,
          n_epoch, ax_loss_train, ax_loss_val,
          kolor, interval_train_loss)
    print('Finished all.')
    plt.pause(1000)
    return

if __name__ == "__main__":
    main()

# https://github.com/Microsoft/CNTK/blob/master/Examples/Image/DataSets/CIFAR-10/install_cifar10.py

from __future__ import print_function
import pickle as cp
import sys
import cifar_utils as ut
import numpy as np
from os.path import join, exists, abspath, basename
from os import makedirs, listdir, getcwd, chdir
from PIL import Image
from torchvision.datasets.utils import check_integrity, download_url

ImgSize = 32
#NumFeat = ImgSize * ImgSize * 3

def saveImage(fname, data):#, label, mapFile, regrFile, pad, **key_parms):
    # data in CIFAR-10 dataset is in CHW format.
    pixData = data.reshape((3, ImgSize, ImgSize))

    img = Image.new('RGB', (ImgSize, ImgSize))
    pixels = img.load()
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            pixels[x, y] = (pixData[0][y][x], pixData[1][y][x], pixData[2][y][x])
    img.save(fname)
    #mapFile.write("%s\t%d\n" % (fname, label))

    # compute per channel mean and store for regression example
    #channelMean = np.mean(pixData, axis=(1, 2))
    #regrFile.write(
    #    "|regrLabels\t%f\t%f\t%f\n" % (channelMean[0] / 255.0, channelMean[1] / 255.0, channelMean[2] / 255.0))
    return



def saveTrainImages(dir_save, li_label, n_im_per_batch, foldername, ext_img):#byte_per_image, ):

    #data = {}
    #dataMean = np.zeros((3, ImgSize, ImgSize))
    dir_train = join(dir_save, foldername)
    i_total = 0
    for ifile in range(1, 6):
        fn_batch = join(join(dir_save, 'cifar-10-batches-py'), 'data_batch_' + str(ifile))
        with open(fn_batch, 'rb') as f:
            if sys.version_info[0] < 3:
                data = cp.load(f)
            else:
                data = cp.load(f, encoding='latin1')
            for i in range(n_im_per_batch):
                i_total += 1
                idx_label = data['labels'][i]
                name_label = li_label[idx_label]
                dir_label = abspath(join(dir_train, name_label))
                if not exists(dir_label):
                    makedirs(dir_label)
                fname = join(dir_label, ('%05d.%s' % (i + (ifile - 1) * 10000, ext_img)))
                saveImage(fname, data['data'][i, :])
                print('Saved %d th image of %s at %s' % (i_total, name_label, fname))

                #saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, regrFile, 4, mean=dataMean)
    #dataMean = dataMean / (50 * 1000)
    #saveMean('CIFAR-10_mean.xml', dataMean)
    return

def saveTestImages(dir_save, li_label, n_im_per_batch, foldername, ext_img):

    #if not os.path.exists(foldername):
        #os.makedirs(foldername)
    dir_test = join(dir_save, foldername)
    fn_batch = join(join(dir_save, 'cifar-10-batches-py'), 'test_batch')
    i_total = 0
    with open(fn_batch, 'rb') as f:
        if sys.version_info[0] < 3:
            data = cp.load(f)
        else:
            data = cp.load(f, encoding='latin1')
        for i in range(n_im_per_batch):
            i_total += 1
            idx_label = data['labels'][i]
            name_label = li_label[idx_label]
            dir_label = abspath(join(dir_test, name_label))
            if not exists(dir_label):
                makedirs(dir_label)
            fname = join(dir_label, ('%05d.%s' % (i, ext_img)))
            saveImage(fname, data['data'][i, :])
            print('Saved %d th image of %s at %s' % (i_total, name_label, fname))


def get_label_names(dir_save):
    fn_meta = join(join(dir_save, 'cifar-10-batches-py'), 'batches.meta')
    batch_meta = cp.load(open(fn_meta, 'rb'))
    n_im_per_batch = batch_meta['num_cases_per_batch']
    li_label = batch_meta['label_names']
    byte_per_image = batch_meta['num_vis']
    return li_label, byte_per_image, n_im_per_batch

def check_if_image_set_exists(dir_save, li_label, n_im_per_label, ext_img):
    does_exist = True
    for label in li_label:
        dir_label = join(dir_save, label)
        if exists(dir_label):
            li_img_file = [file for file in listdir(dir_label) if file.endswith(ext_img)]
            n_img = len(li_img_file)
            if n_im_per_label != n_img:
                does_exist = False
                break
        else:
            does_exist = False
            break
    return does_exist


def check_if_download_done(dir_save):

    if not check_if_uncompression_done(dir_save):
        filename = "cifar-10-python.tar.gz"
        fn = join(dir_save, filename)
        return exists(fn)
    return True

def check_if_uncompression_done(dir_save):

    base_folder = 'cifar-10-batches-py'

    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    root = dir_save
    for fentry in (train_list + test_list):
        filename, md5 = fentry[0], fentry[1]
        fpath = join(root, base_folder, filename)
        if not check_integrity(fpath, md5):
            return False
    return True




def prepare_cifar10_dataset(dir_save, ext_img):

    #dir_save = './data'
    n_im_per_label_train, n_im_per_label_test = 5000, 1000
    foldername_train, foldername_test = 'train', 'test'
    if not check_if_download_done(dir_save):
        import tarfile
        url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        filename = "cifar-10-python.tar.gz"
        tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
        root = dir_save
        download_url(url, root, filename, tgz_md5)
    if not check_if_uncompression_done(dir_save):
        # extract file
        cwd = getcwd()
        tar = tarfile.open(join(root, filename), "r:gz")
        chdir(root)
        tar.extractall()
        tar.close()
        chdir(cwd)
        #loadData(url, dir_save)

    li_label, byte_per_image, n_im_per_batch = get_label_names(dir_save)
    dir_train, dir_test = join(dir_save, foldername_train), join(dir_save, foldername_test)
    if check_if_image_set_exists(
            dir_train, li_label, n_im_per_label_train, ext_img):
        print(dir_train + ' are already existing.')
    else:
        saveTrainImages(dir_save, li_label, n_im_per_batch, foldername_train, ext_img)
    if check_if_image_set_exists(
            dir_test, li_label, n_im_per_label_test, ext_img):
        print(dir_test + ' are already existing.')
    else:
        saveTestImages(dir_save, li_label, n_im_per_batch, foldername_test, ext_img)
    return li_label

# https://github.com/Microsoft/CNTK/blob/master/Examples/Image/DataSets/CIFAR-10/install_cifar10.py

from __future__ import print_function
import pickle as cp
import sys
import cifar_utils as ut
import numpy as np
from os.path import join, exists, abspath, basename
from os import makedirs, listdir
from PIL import Image

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



def saveTrainImages(dir_save, li_label, n_im_per_batch, foldername)#byte_per_image, ):

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
                fname = join(dir_label, ('%05d.png' % (i + (ifile - 1) * 10000)))
                saveImage(fname, data['data'][i, :])
                print('Saved %d th image of %s at %s' % (i_total, name_label, fname))

                #saveImage(fname, data['data'][i, :], data['labels'][i], mapFile, regrFile, 4, mean=dataMean)
    #dataMean = dataMean / (50 * 1000)
    #saveMean('CIFAR-10_mean.xml', dataMean)
    return

def saveTestImages(dir_save, li_label, n_im_per_batch, foldername):

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
            fname = join(dir_label, ('%05d.png' % i))
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
        li_img_file = [file for file in listdir(dir_label) if file.endswith(ext_img)]
        n_img = len(li_img_file)
        if n_im_per_label != n_img:
            does_exist = False
            break
    return does_exist






def prepare_cifar10_dataset(dir_save, ext_img):
#if __name__ == "__main__":
    #dir_save = './data'
    n_im_per_label_train, n_im_per_label_test = 5000, 1000
    foldername_train, foldername_test = 'train', 'test'

    if not check_if_download_done(dir_save):
        loadData('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                       dir_save)

    li_label, byte_per_image, n_im_per_batch = get_label_names(dir_save)
    if not check_if_image_set_exists(dir_save, li_label, n_im_per_label_train, ext_img):
        saveTrainImages(dir_save, li_label, byte_per_image, n_im_per_batch, foldername_train)
    if not check_if_image_set_exists(dir_save, li_label, n_im_per_label_test, ext_img):
        saveTestImages(dir_save, li_label, byte_per_image, n_im_per_batch, foldername_test)
    path_train_txt = join(dir_save, r'Train_cntk_text.txt')
    path_test_txt = join(dir_save, r'Test_cntk_text.txt')
    if not (exists(path_train_txt) and exists(path_test_txt)):
        trn, tst= ut.loadData('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                              dir_save)
        print ('Writing train text file...')
        ut.saveTxt(path_train_txt, trn)
        print ('Done.')
        print ('Writing test text file...')
        ut.saveTxt(path_test_txt, tst)
        print ('Done.')
    else:
        print(path_train_txt + ' and ' + path_test_txt + ' are already existing.')
    folder_train, folder_test = 'train', 'test'
    dir_train_img = join(dir_save, folder_train)
    dir_test_img = join(dir_save, folder_test)
    if not (exists(dir_train_img) and exists(dir_test_img)):
        print ('Converting train data to png images...')
        ut.saveTrainImages(path_train_txt, dir_train_img, dir_save)
        print ('Done.')
        print ('Converting test data to png images...')
        ut.saveTestImages(path_test_txt, dir_test_img, dir_save)
        print ('Done.')
    else:
        print(dir_train_img + ' and ' + dir_test_img + ' are already existing.')
        print ('There is nothing to do. The traing and test image files are already made.')

# https://github.com/Microsoft/CNTK/blob/master/Examples/Image/DataSets/CIFAR-10/install_cifar10.py

from __future__ import print_function
import cifar_utils as ut
from os.path import join, exists, basename

if __name__ == "__main__":
    dir_save = './data'
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

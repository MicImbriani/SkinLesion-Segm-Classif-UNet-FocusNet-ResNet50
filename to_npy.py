from os import listdir
from os.path import splitext
from PIL import Image

import cv2
import numpy as np

def turn_npy_imgs(folder_path):    
    images = [splitext(file)[0] for file in listdir(folder_path)]
    imgs_array = []
    for image in sorted(images):
        path = folder_path + "/" + image + ".png"
        im = cv2.imread(path, 0)
        im = np.asarray(im, dtype=np.float32)
        im = im[:,:,np.newaxis]
        im = im.tolist()
        imgs_array.append(im)
    npa = np.asarray(imgs_array, dtype=np.float32)
    print(npa.shape)
    return npa

def turn_npy_masks(folder_path):  
    images = [splitext(file)[0] for file in listdir(folder_path)]
    imgs_array = []
    imgs_array1 = []
    for image in sorted(images):
        path = folder_path + "/" + image + ".png"
        im = cv2.imread(path, 0)
        ret, im1 = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
        ret2, im2 = cv2.threshold(im, 127, 1, cv2.THRESH_BINARY)
        im1 = np.asarray(im1, dtype=np.float32)
        im1 = im1[:,:,np.newaxis]
        im1 = im1.tolist()
        im2 = im2.tolist()
        imgs_array.append(im1)
        imgs_array1.append(im2)
    npa = np.asarray(imgs_array, dtype=np.float32)
    npa1 = np.asarray(imgs_array1, dtype=np.float32)
    print(npa.shape)
    return npa, npa1

def turn_npy(path, save_path):
    images_folder_path = path + "/ISIC-2017_Training_Data"
    masks_folder_path = path + "/ISIC-2017_Training_Part1_GroundTruth"
    val_imgs_folder_path = path + "/ISIC-2017_Validation_Data"
    val_masks_folder_path = path + "/ISIC-2017_Validation_Part1_GroundTruth"
    test_imgs_folder_path = path + "/ISIC-2017_Test_v2_Data"
    test_masks_folder_path = path + "/ISIC-2017_Test_v2_Part1_GroundTruth"

    # train_X = turn_npy_imgs(images_folder_path)
    # train_y, train_y1 = turn_npy_masks(masks_folder_path)

    # val_X = turn_npy_imgs(val_imgs_folder_path)
    val_y, val_y1 = turn_npy_masks(val_masks_folder_path)

    test_X = turn_npy_imgs(test_imgs_folder_path)
    test_y, test_y1 = turn_npy_masks(test_masks_folder_path)

    ###############


    # train_X = train_X.astype('float32')
    # val_X = val_X.astype('float32')
    # test_X = test_X.astype('float32')

    # mean = 153.55293  # mean for data centering
    # std = 41.8674  # std for data normalization

    # # train_X -= mean
    # # train_X /= std

    # # val_X -= mean 
    # # val_X /= std

    # test_X /= std
    # test_X -= mean 

    # train_y = train_y.astype('float32')
    # train_y /= 255.  # scale masks to [0, 1]

    # val_y = val_y.astype('float32')
    # val_y /= 255.  # scale masks to [0, 1]

    # test_y = test_y.astype('float32')
    # test_y /= 255.  # scale masks to [0, 1]


    # np.save(save_path + '/data.npy',train_X)
    # np.save(save_path + '/dataMask.npy',train_y)
    # np.save(save_path + '/dataval.npy', val_X)
    np.save(save_path + '/dataMaskval.npy', val_y)
    np.save(save_path + '/datatest.npy', test_X)
    np.save(save_path + '/dataMasktest.npy', test_y)

    
path = "/var/tmp/mi714/NEW/aug_dataset"
save_path = "/var/tmp/mi714/NEW/npy_dataset"
turn_npy(path, save_path)
from os import listdir
from os.path import splitext

from PIL import Image
import cv2
import numpy as np

def turn_npy_imgs(folder_path):    
    """Turns images to numpy arrays and returns them.

    Args:
        folder_path (string): Path to images folder.

    Returns:
        npa (numpy array): Numpy array containing the images.
    """    
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
    """Turn masks to numpy arrays, treshold to 0 or 255 and returns them.

    Args:
        folder_path (string): Path to masks folder.

    Returns:
        npa (numpy array): Numpy array containing the masks.
    """    
    images = [splitext(file)[0] for file in listdir(folder_path)]
    imgs_array = []
    for image in sorted(images):
        path = folder_path + "/" + image + ".png"
        im = cv2.imread(path, 0)
        ret, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
        im = np.asarray(im, dtype=np.float32)
        im = im[:,:,np.newaxis]
        im = im.tolist()
        imgs_array.append(im)
    npa = np.asarray(imgs_array, dtype=np.float32)
    print(npa.shape)
    return npa 

def turn_npy(path, save_path):
    """Takes root path containing splits of dataset, 
    converts to numpy and saves in .npy format.

    Args:
        path (string): Path to dataset.
        save_path (string): Path to destination folder to save .npy files.
    """    
    images_folder_path = path + "/ISIC-2017_Training_Data"
    masks_folder_path = path + "/ISIC-2017_Training_Part1_GroundTruth"
    val_imgs_folder_path = path + "/ISIC-2017_Validation_Data"
    val_masks_folder_path = path + "/ISIC-2017_Validation_Part1_GroundTruth"
    test_imgs_folder_path = path + "/ISIC-2017_Test_v2_Data"
    test_masks_folder_path = path + "/ISIC-2017_Test_v2_Part1_GroundTruth"

    train_X = turn_npy_imgs(images_folder_path)
    train_y, train_y1 = turn_npy_masks(masks_folder_path)

    val_X = turn_npy_imgs(val_imgs_folder_path)
    val_y, val_y1 = turn_npy_masks(val_masks_folder_path)

    test_X = turn_npy_imgs(test_imgs_folder_path)
    test_y, test_y1 = turn_npy_masks(test_masks_folder_path)


    np.save(save_path + '/data.npy',train_X)
    np.save(save_path + '/dataMask.npy',train_y)
    np.save(save_path + '/dataval.npy', val_X)
    np.save(save_path + '/dataMaskval.npy', val_y)
    np.save(save_path + '/datatest.npy', test_X)
    np.save(save_path + '/dataMasktest.npy', test_y)



def turn_npy_nomasks(path, save_path):
    """Takes root path containing splits of dataset, 
    converts to numpy and saves in .npy format.

    Args:
        path (string): Path to dataset.
        save_path (string): Path to destination folder to save .npy files.
    """    
    images_folder_path = path + "/ISIC-2017_Training_Data"
    val_imgs_folder_path = path + "/ISIC-2017_Validation_Data"
    test_imgs_folder_path = path + "/ISIC-2017_Test_v2_Data"

    train_X = turn_npy_imgs(images_folder_path)
    val_X = turn_npy_imgs(val_imgs_folder_path)
    test_X = turn_npy_imgs(test_imgs_folder_path)

    np.save(save_path + '/data.npy',train_X)
    np.save(save_path + '/dataval.npy', val_X)
    np.save(save_path + '/datatest.npy', test_X)

turn_npy_nomasks("/var/tmp/mi714/NEW/cropped_datasets/unet", "/var/tmp/mi714/NEW/npy_dataset_cropped/unet")
turn_npy_nomasks("/var/tmp/mi714/NEW/cropped_datasets/unet_bn", "/var/tmp/mi714/NEW/npy_dataset_cropped/unet_bn")
turn_npy_nomasks("/var/tmp/mi714/NEW/cropped_datasets/unet_res_se", "/var/tmp/mi714/NEW/npy_dataset_cropped/unet_res_se")
turn_npy_nomasks("/var/tmp/mi714/NEW/cropped_datasets/focusnet", "/var/tmp/mi714/NEW/npy_dataset_cropped/focusnet")
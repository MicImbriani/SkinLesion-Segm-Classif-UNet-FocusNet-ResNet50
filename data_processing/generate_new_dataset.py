import os
from os.path import splitext
from os import listdir
from PIL import Image
import cv2

import numpy as np
from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical

from data_process import get_result
import sys
sys.path.append('/home/userfs/m/mi714/Desktop/Keras-PRBX/networks')
from unet_nn import unet
from unet_res_se_nn import unet_res_se
from focus import get_focusnetAlpha
from resnet import get_res





def generate_masks(model, path, save_path):
    """Takes a model as input and predicts segmentation masks for each image.

    Args:
        model (Keras model): Compiled model to use for generating predictions.
        path (string): Path to folder with images to be predicted.
        save_path (string): Path to destination folder to save predicted masks.
    """    
    # trainData = np.load('/var/tmp/mi714/NEW/npy_dataset/data.npy')
    # trainMask = np.load('/var/tmp/mi714/NEW/npy_dataset/dataMask.npy')

    # valData = np.load('/var/tmp/mi714/NEW/npy_dataset/dataval.npy')
    # valMask = np.load('/var/tmp/mi714/NEW/npy_dataset/dataMaskval.npy')

    testData = np.load('/var/tmp/mi714/NEW/npy_dataset/datatest.npy')
    testMask = np.load('/var/tmp/mi714/NEW/npy_dataset/dataMasktest.npy')

    #p = focus.predict(trainData)
    images = [file for file in listdir(path)]
    #files = os.listdir(path)
    for image in images:
        img_id = splitext(image)[0]
        x = Image.open(path + "/" + image).convert('L')
        x = np.asarray(x, dtype=np.float32)
        x = x[np.newaxis,...]   
        x = x[...,np.newaxis]
        x = np.asarray(x)
        img = model.predict(x)
        img = img[0,:,:,:]
        _, img = cv2.threshold(img, 0.5, 255, cv2.THRESH_BINARY)
        img = np.asarray(img, dtype=np.float32)
        img = Image.fromarray(img)
        img = img.convert("L")
        img.save(save_path + "/" + image)




def generate_dataset(pred_masks_path, images_path, save_path):
    """Generate new datasets to be used for training the ResNets
    on the skin lesion classification task.
    For each mask, the respective original images is retrieved;
    the mask is then thresholded to 1, and multiplied with the image.

    Args:
        pred_masks_path (string): Path to directory containing masks.
        images_path (string): Path to directory containing images.
        save_path (string): Path to destination folder in which new 
                            cropped images will be saved.
    """     
    pred_masks = listdir(pred_masks_path)
    for mask in pred_masks:
        mask_id = mask 
        image_id = mask_id.replace("_segmentation", "")
        
        mask = Image.open(pred_masks_path+ "/" + mask_id).convert('L')
        image = Image.open(images_path + "/" + image_id).convert('L')
        mask = np.asarray(mask, dtype=np.float32)
        image = np.asarray(image, dtype=np.float32)

        _, mask = cv2.threshold(mask, 127, 1., cv2.THRESH_BINARY)
        crop = image * mask 
        crop = Image.fromarray(crop)
        crop = crop.convert("L")
        crop.save(save_path + "/" + image_id)



def generate_targets(path, csv_path):
        images = os.listdir(path)
        images.sort()
        labels = []

        for image in images:
                image_id = os.path.splitext(image)[0]
                temp = get_result(image_id, csv_path)
                labels.append(temp)
        labels = np.array(labels)
        labels = labels[:, np.newaxis]

        y_array = to_categorical(labels, 2)
        y_array = np.array(y_array)
        return y_array 




if __name__ == "__main__":
    ############## MODELS ##############
    # SEGMENTATION
    # U-Net
    # model = unet(batch_norm=False)
    # model.load_weights("/var/tmp/mi714/NEW/models/UNET/unet8/unet8_weights.h5")
    # U-Net BatchNorm
    # model = unet(batch_norm=True)
    # model.load_weights("/var/tmp/mi714/NEW/models/UNET_BN/unet_bn3/unet_bn3_weights.h5")
    # U-Net Res SE
    # model = unet_res_se()
    # model.load_weights("/var/tmp/mi714/NEW/models/UNET_RES_SE/unet_res_se3/unet_res_se3_weights.h5")
    #Focusnet
    model = get_focusnetAlpha()
    model.load_weights("/var/tmp/mi714/NEW/models/FOCUS/focusnet5/focusnet5_weights.h5")

    # CLASSIFICATION
    # ResNet OG
    # model = get_res()
    # model.load_weights("/var/tmp/mi714/NEW/models/RESNETS/RESNET_OG/resnet_og/resnet_weights.h5")



    ############## PREDICTED MASKS GENERATION ##############
    # dataset_path = "/var/tmp/mi714/NEW/aug_dataset"

    # save_path = "/var/tmp/mi714/NEW/predictions/unet_res_se"
    # train_save = save_path + "/ISIC-2017_Training_Data"
    # val_save = save_path + "/ISIC-2017_Validation_Data"
    # test_save = save_path + "/ISIC-2017_Test_v2_Data"

    # os.makedirs(train_save, exist_ok=True)
    # os.makedirs(val_save, exist_ok=True)
    # os.makedirs(test_save, exist_ok=True)
    
    # # Train set
    # generate_masks(model,
    #              dataset_path + "/ISIC-2017_Training_Data",
    #              train_save)
    
    # Validation set
    # generate_masks(model,
    #              dataset_path + "/ISIC-2017_Validation_Data",
    #              val_save)
    
    # Test set
    # generate_masks(model,
    #              dataset_path + "/ISIC-2017_Test_v2_Data",
    #              test_save)
    


    ############# GENERATION OF NEW DATASETS (image * predicted masks) ##############
    dataset_path = "/var/tmp/mi714/NEW/aug_dataset"
    train_path = dataset_path + "/ISIC-2017_Training_Data"
    val_path = dataset_path + "/ISIC-2017_Validation_Data"
    test_path = dataset_path + "/ISIC-2017_Test_v2_Data"

    pred_masks = "/var/tmp/mi714/NEW/predictions/unet_bn"       ######################### <== change model here
    train_pred_masks = pred_masks + "/ISIC-2017_Training_Data"
    val_pred_masks = pred_masks + "/ISIC-2017_Validation_Data"
    test_pred_masks = pred_masks + "/ISIC-2017_Test_v2_Data"

    save_path = "/var/tmp/mi714/NEW/cropped_datasets/unet_bn"   ######################### <== change model here
    train_save = save_path + "/ISIC-2017_Training_Data"
    val_save = save_path + "/ISIC-2017_Validation_Data"
    test_save = save_path + "/ISIC-2017_Test_v2_Data"

    os.makedirs(train_save, exist_ok=True)
    os.makedirs(val_save, exist_ok=True)
    os.makedirs(test_save, exist_ok=True)

    # Train set
    generate_dataset(train_pred_masks, train_path, train_save)
    # Validation set
    generate_dataset(val_pred_masks, val_path, val_save)
    # Test set
    generate_dataset(test_pred_masks, test_path, test_save)
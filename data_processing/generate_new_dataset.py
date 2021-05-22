import os, glob
import cv2
#from keras.backend.cntk_backend import dtype
import numpy as np
from keras.preprocessing.image import img_to_array
from os.path import splitext
from os import listdir
from PIL import Image
# from networks.unet_nn import unet
# from networks.unet_res_se_nn import unet_res_se
# from networks.focus import get_focusnetAlpha




def divide_imgs_by_class(path):
    # path = "/var/tmp/mi714/class_division"
    train_path = path + "/Train"
    val_path = path + "/Validation"
    test_path = path + "/Test"

    train_no_mel_path = train_path + "/no_melanoma"
    train_mel_path = train_path + "/melanoma"

    val_no_mel_path = val_path + "/no_melanoma"
    val_mel_path = val_path + "/melanoma"

    test_no_mel_path = test_path + "/no_melanoma"
    test_mel_path = test_path + "/melanoma"


    sets = ["Train", "Validation", "Test"]
    for set in sets:
        images = [file for file in listdir(path +"/"+set) if not "melanoma" in file]
        csv_path = path + "/"+set +"_GT_result.csv"
        for image in sorted(images):
            image_id = splitext(image)[0]
            res = get_result(image_id, csv_path)
            if res == 0:
                savepath = path + "/" + set + "/no_melanoma"
            else:
                savepath = path + "/" + set + "/melanoma"
            os.makedirs(savepath, exist_ok=True)
            image_path = path + "/" +set + "/" + image
            img = Image.open(image_path)
            img.save(savepath+ "/"+image)





# path is the path with actual images
def generate_masks(model, path, save_path):
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




if __name__ == "__main__":
    #unet = unet((256,256,1), batch_norm=False)
    # unetresse = get_unet()
    # focus = focusnet()
    # model = get_focusnetAlpha()

    # #unet.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/unet_nobn/unet_nobn1/unet_nobn_weights.h5")
    # #unetbn.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/unet_bn/unet_bn/unet_bn_weights.h5")
    # #unetresse.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/unet_polished/unet_polished_weights.h5")
    # #focus.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/focusnet99/focus_weights.h5")
    # model.load_weights("/var/tmp/mi714/NEW/models/focusnet_dice/focusnet_dice_weights.h5")

    # dataset_path = "/var/tmp/mi714/NEW/aug_dataset/ISIC-2017_Test_v2_Data"
    # save_path = "/var/tmp/mi714/NEW/predictions/focusnet/test"
    # generate_new(model, dataset_path, save_path)


    generate_dataset(pred_masks_path="D:/Users/imbrm/ISIC_2017_new/small/ISIC-2017_Training_Part1_GroundTruth",
                    images_path="D:/Users/imbrm/ISIC_2017_new/small/ISIC-2017_Training_Data",
                    save_path="D:/Users/imbrm/ISIC_2017_new/small/delete")


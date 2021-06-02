import os
from os.path import splitext
from os import listdir
import random
import csv

import cv2
from PIL import Image, ImageFile
import albumentations as A
import pandas as pd
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed

from data_processing.data_process import grey_resize, get_result



# Make PIL tolerant of uneven images block sizes.
ImageFile.LOAD_TRUCATED_IMAGES = True



def augment_operations(image_id, image_folder_path, mask_folder_path, train_val, x):
    """Perform augmentation operations on the inputted image.
    Seed is used for for applying the same augmentation to the image and its mask.

    Args:
        image_id (string): The ID of the image to be augmented.
        image_folder_path (string): Path of folder in which the augmented img will be saved.
        mask_folder_path (string): Path of folder in which the augmented mask will be saved.
        train_val (string): Specifies whether it's "Train" or "Validation".
        x (string): The suffix to be added to images IDs for augmentated images.

    Returns:
        new_img (Image): New augmented PIL image.
        new_img_mask (Image): New augmented PIL mask.
    """
    mask_id = image_id + "_segmentation"

    image = cv2.imread(image_folder_path + "/" + image_id + ".png", 0)
    mask = cv2.imread(mask_folder_path + "/" + mask_id + ".png", 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    

    transform = A.Compose([
        A.ElasticTransform(alpha=30, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.CLAHE(p=1),
        A.GaussNoise(p=1),
        A.Resize(256,256),
        A.ToGray(p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomRotate90(p=1),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
    ])

    image_id = image_id + x

    ##############################  <--- Comment out HERE to load seeds from .csv file 
    # Set random seed.
    seed = np.random.randint(0, 2**30)

    if train_val == "Validation":
        # Write seed in .csv file
        with open('seedval.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([image_id, seed])
    else:
        # Write seed in .csv file
        with open('seeds.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([image_id, seed])
    ##############################
    

    ############################## <--- Uncomment HERE to load seeds from .csv file 
    # # Load seed from file
    # if train_val == "Validation":
    #     df = pd.read_csv("seedval.csv")
    # else:
    #     df = pd.read_csv("seed.csv")

    # img_index = df.loc[df["image_id"] == image_id].index[0]
    # seed = df.at[img_index, "seed"]
    ##############################

    # Set random seed
    random.seed(seed)

    # Transform the image and mask using the same transformation.
    transformed = transform(image=image, mask=mask)
    new_img = transformed['image']
    new_img_mask = transformed['mask']

    new_img = Image.fromarray(new_img)
    new_img_mask = Image.fromarray(new_img_mask)

    return new_img, new_img_mask


def augment_img(image_id, images_folder_path, masks_folder_path, csv_file_path, train_val):
    """Executes augmentation on a single image and mask, saves them, and turn image to greyscale. 
    If mole is melanoma, perform 4 augmentation with probability=1.
    It performs the same transformation on the image and its relative mask.

    Args:
        image_id (string): ID of the image to be augmented.
        images_folder_path (string): Path of folder in which the augmented img will be saved.
        masks_folder_path (string): Path of folder in which the augmented mask will be saved.
        csv_file_path (string): Path leading to the .csv file with ground truth.
        train_val (string): Specifies whether it's "Train" or "Validation".
    """
    # If the image is from Validation set, always perform aumgmentation 
    if train_val == "Validation":
        img_1, img_1_mask = augment_operations(image_id, images_folder_path, masks_folder_path, train_val, "")

        # Save image and mask in two dedicated folders.
        img_1.save(images_folder_path + "/" + image_id + "x1" + ".png", "PNG", quality=100)
        img_1_mask.save(masks_folder_path + "/" + image_id + "_segmentation" + "x1" + ".png", "PNG", quality=100)
        
        grey_resize(image_id, images_folder_path, masks_folder_path)
        
        return
    
    # If image is from train set perform 4 sugmentations only if it's melanoma.
    else:
        melanoma = int(get_result(image_id, csv_file_path))

        if melanoma == 1:
            # Perform augmentations, store the resulting images and masks.
            img_1, img_1_mask = augment_operations(image_id, images_folder_path, masks_folder_path, train_val, "x1")
            img_2, img_2_mask = augment_operations(image_id, images_folder_path, masks_folder_path, train_val, "x2")
            img_3, img_3_mask = augment_operations(image_id, images_folder_path, masks_folder_path, train_val, "x3")
            img_4, img_4_mask = augment_operations(image_id, images_folder_path, masks_folder_path, train_val, "x4")

            # Save images in dedicated folder.
            img_1.save(images_folder_path + "/" + image_id + "x1" + ".png", "PNG", quality=100)
            img_2.save(images_folder_path + "/" + image_id + "x2" + ".png", "PNG", quality=100)
            img_3.save(images_folder_path + "/" + image_id + "x3" + ".png", "PNG", quality=100)
            img_4.save(images_folder_path + "/" + image_id + "x4" + ".png", "PNG", quality=100)

            # Save masks in dedicated folder.
            img_1_mask.save(masks_folder_path + "/" + image_id + "_segmentation" + "x1" + ".png", "PNG", quality=100)
            img_2_mask.save(masks_folder_path + "/" + image_id + "_segmentation" + "x2" + ".png", "PNG", quality=100)
            img_3_mask.save(masks_folder_path + "/" + image_id + "_segmentation" + "x3" + ".png", "PNG", quality=100)   
            img_4_mask.save(masks_folder_path + "/" + image_id + "_segmentation" + "x4" + ".png", "PNG", quality=100)

            # Add new datapoint to .csv file 
            with open(csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([image_id + "x1", 1, 0])
                writer.writerow([image_id + "x2", 1, 0])
                writer.writerow([image_id + "x3", 1, 0])
                writer.writerow([image_id + "x4", 1, 0])
        
        # Resize and turn to grey regardless of diagnosis result
        grey_resize(image_id, images_folder_path, masks_folder_path)


def augment_dataset(images_folder_path, masks_folder_path, csv_file_path, jobs, train_val):
    """Performs augmentation on the whole dataset.
    Augmentation is performed in parallel to speed up process.

    Args:
        images_folder_path (string): Path to folder containing images of moles.
        masks_folder_path (string): Path to folder containing images of masks.
        csv_file_path (string): Path to .csv file containing ground truth.
        jobs (int): Number by which the parallelisation will be applied concurrently.
        train_val (string): Specifies whether it's "Train" or "Validation".
    """
    images = [splitext(file)[0] for file in listdir(images_folder_path)]
    print(f"Augmenting {train_val} Images and Masks:")
    Parallel(n_jobs=jobs)(
        delayed(augment_img)(
            image, images_folder_path, masks_folder_path, csv_file_path, train_val
        )
        for image in tqdm(images)
    )

############################################################################################
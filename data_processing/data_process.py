import os
from os.path import splitext
from os import listdir
import random
import cv2
import csv

import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed
import albumentations as A


# Make PIL tolerant of uneven images block sizes.
ImageFile.LOAD_TRUCATED_IMAGES = True


def del_superpixels(input_path, jobs):
    """Deletes the superpixels images of the skin lesions.

    Args:
        input_path (string): Path of the folder containing the superpixel images.
        jobs (string): Number of jobs to be used for parallelisation.
    """
    # Store the IDs of all the _superpixel images in a list.
    images = [
        splitext(file)[0]
        for file in listdir(input_path)
        if "_superpixels" in splitext(file)[0]
    ]
    print("Deleting Superpixel Images:")
    Parallel(n_jobs=jobs)(
        delayed(os.remove)(str(input_path + "/" + str(image + ".png")))
        for image in tqdm(images)
    )


def grey_resize(image_id, images_folder_path, masks_folder_path):
    image = cv2.imread(images_folder_path + "/" + image_id + ".png", 0)
    mask = cv2.imread(masks_folder_path + "/" + image_id + "_segmentation" + ".png", 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    transform = A.Compose([A.Resize(256, 256), A.ToGray(p=1),])

    transformed = transform(image=image, mask=mask)
    new_img = transformed["image"]
    new_img_mask = transformed["mask"]

    new_img = Image.fromarray(new_img)
    new_img_mask = Image.fromarray(new_img_mask)

    new_img.save(images_folder_path + "/" + image_id + ".png", "PNG", quality=100)
    new_img_mask.save(
        masks_folder_path + "/" + image_id + "_segmentation" + ".png",
        "PNG",
        quality=100,
    )


def get_result(image_id, csv_file_path):
    """Checks whether the inputted image was a melanoma or not.

    Args:
        image_id (string): ID of the image.
        csv_file_path (string): Path leading to the .csv file with ground truth.

    Returns:
        melanoma (int): The melanoma classification result in 0 or 1.
    """
    df = pd.read_csv(csv_file_path)
    img_index = df.loc[df["image_id"] == image_id].index[0]
    melanoma = df.at[img_index, "melanoma"]
    return melanoma


def convert(image, folder):
    """Parallelisable function for converting all the images from JPEG to PNG format.

    Args:
        image (string): Image ID to convert.
        folder (string): Path of the folder containing images to convert.
    """
    try:
        img = Image.open(folder + "/" + image + ".jpg")
        img.save(folder + "/" + image + ".png")
        os.remove(folder + "/" + image + ".jpg")
    except:
        pass


def convert_format(folder, jobs, train_or_val):
    """Converts all the images from JPEG to PNG format.

    Args:
        folder (string): Path of the folder containing images to convert.
        jobs (int): Number of jobs for parallelisation.
        train_or_val (string): Specifies whether it's Train or Validation images.
    """
    images = [splitext(file)[0] for file in listdir(folder)]
    print(f"Converting {train_or_val} from JPEG to PNG.")
    Parallel(n_jobs=jobs)(delayed(convert)(image, folder) for image in tqdm(images))

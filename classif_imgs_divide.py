import pandas as pd
import os
from PIL import Image
from os import listdir
from os.path import splitext
from data_processing.data_process import get_result



# Need dataset to be structured like in /var/tmp/mi714/NEW/aug_dataset
def separate_dataset():
    # Augmented dataset
    path = "/var/tmp/mi714/NEW/aug_dataset"

    train_path = path + "/ISIC-2017_Training_Data"
    train_GT = path + "/ISIC-2017_Training_Part1_GroundTruth"

    val_path = path + "/ISIC-2017_Validation_Data"
    val_GT = path + "/ISIC-2017_Validation_Part1_GroundTruth"

    test_path = path + "/ISIC-2017_Test_v2_Data"
    test_GT = path + "/ISIC-2017_Test_v2_Part1_GroundTruth"



    # Dataset divided in melanoma/no-melanoma folders
    new_path = "/var/tmp/mi714/NEW/classif_dataset"

    imgs_mel = "/Mel"
    imgs_nomel = "/NoMel"

    masks_mel = "/Mel"
    masks_nomel = "/NoMel"

    sets = [("ISIC-2017_Training_Data",
            "ISIC-2017_Training_Part1_GroundTruth",
            "ISIC-2017_Training_Part3_GroundTruth"
            ),
            ("ISIC-2017_Validation_Data",
            "ISIC-2017_Validation_Part1_GroundTruth",
            "ISIC-2017_Validation_Part3_GroundTruth"
            ),
            ("ISIC-2017_Test_v2_Data",
            "ISIC-2017_Test_v2_Part1_GroundTruth",
            "ISIC-2017_Test_v2_Part3_GroundTruth"
            )]



    for images_dir_name, masks_dir_name, csv_path in sets:
        images_dir = [file for file in listdir(path + "/" + images_dir_name) if not "melanoma" in file]
        masks_dir = [file for file in listdir(path + "/" + masks_dir_name) if not "melanoma" in file]
        csv_path = path + "/" + csv_path + ".csv"

        for image, mask in zip(sorted(images_dir), sorted(masks_dir)):
            print(image + "    " + mask)
            assert image[:11] == mask[:11]
            image_id = splitext(image)[0]
            res = get_result(image_id, csv_path)
            if res == 0:
                img_savepath = new_path + "/" + images_dir_name + "/" + imgs_nomel
                mask_savepath = new_path + "/" + masks_dir_name + "/" + masks_nomel
            else:
                img_savepath = new_path + "/" + images_dir_name + "/" + imgs_mel
                mask_savepath = new_path + "/" + masks_dir_name + "/" + masks_mel
            os.makedirs(img_savepath, exist_ok=True)
            os.makedirs(mask_savepath, exist_ok=True)
            img = Image.open(path + "/" + images_dir_name + "/" + image)
            gt = Image.open(path + "/" + masks_dir_name + "/" + mask)
            img.save(img_savepath + "/" + image)
            gt.save(mask_savepath + "/" + mask)


if __name__ == "__main__":
    separate_dataset()
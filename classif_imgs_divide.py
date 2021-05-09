import pandas as pd
import os
from PIL import Image
from os import listdir
from os.path import splitext


def get_result(image_id, csv_file_path):
    """Checks whether the inputted image was a melanoma or not.

    Args:
        image_id (string): ID of the image.
        csv_file_path (string): Path leading to the .csv file with ground truth.

    Returns:
        melanoma (int): The melanoma classification result in 0 or 1.
    """
    df = pd.read_csv(csv_file_path)
    print(image_id)
    img_index = df.loc[df["image_id"] == image_id].index[0]
    melanoma = df.at[img_index, "melanoma"]
    return melanoma


path = "/var/tmp/mi714/class_division"
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
    
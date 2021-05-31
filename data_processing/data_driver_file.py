import os
from os.path import splitext
from os import listdir
import csv

from data_processing.to_npy import turn_npy
from data_processing.augmentation import augment_dataset
from data_processing.data_process import del_superpixels, convert_format, grey_resize



def generate_dataset(path, n_jobs):
    """Generate augmented dataset.

    Args:
        path (string): Path to original ISIC dataset.
        n_jobs (int): Number of jobs to use for parallelisation.
    """    
    # Generate the Train paths.
    images_folder_path = path + "/ISIC-2017_Training_Data"
    masks_folder_path = path + "/ISIC-2017_Training_Part1_GroundTruth"
    csv_file_path = path + "/Train_GT_result.csv"

    # Delete superpixels.
    del_superpixels(images_folder_path, n_jobs)

    # Convert JPEG -> PNG
    convert_format(images_folder_path, 8, "Train")

    # Delete metadata file.
    try:
        os.remove(images_folder_path + "/ISIC-2017_Training_Data_metadata.csv")
    except: 
        pass

    # Create new .csv file with seeds 
    with open('seeds.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "seed"])

    # Augment with relative masks.
    augment_dataset(
        images_folder_path,
        masks_folder_path,
        csv_file_path,
        n_jobs,
        "Train"
    )

    print("Training Complete.")


    ######################
    # VALIDATION 
    
    # Generate Validation paths.
    valimages_folder_path = path + "/ISIC-2017_Validation_Data"
    valmasks_folder_path = path + "/ISIC-2017_Validation_Part1_GroundTruth"
    csv_file_path = path + "/Validation_GT_result"

    # Delete superpixels.
    del_superpixels(valimages_folder_path, n_jobs)

    # Convert JPEG -> PNG
    convert_format(valimages_folder_path, 8, "Validation")

    # Delete metadata file.
    try:
        os.remove(valimages_folder_path + "/" + "ISIC-2017_Validation_Data_metadata.csv")
    except: 
        pass

    # Create new .csv file with seeds for validation data
    with open('seedval.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "seed"])

    # Augment with relative masks.
    augment_dataset(
        valimages_folder_path,
        valmasks_folder_path,
        csv_file_path,
        n_jobs,
        "Validation"
    )

    print("Validation Complete.")


    ######################
    # TEST
    images_folder_path = path + "/ISIC-2017_Test_v2_Data"
    masks_folder_path = path + "/ISIC-2017_Test_v2_Part1_GroundTruth"

    # Delete superpixels.
    del_superpixels(images_folder_path, n_jobs)

    # Delete metadata file.
    try:
        os.remove(images_folder_path + "/" + "ISIC-2017_Test_v2_Data_metadata.csv")
    except: 
        pass

    # Convert JPEG to PNG
    convert_format(images_folder_path, 8, "Test")

    images = listdir(images_folder_path)
    for image_id in images:
        image_id = splitext(image_id)[0]
        grey_resize(image_id, images_folder_path, masks_folder_path)

    print("Test Complete.")





if __name__ == "__main__":
    path = "D:/Users/imbrm/ISIC_2017_new"

    generate_dataset(path, 5)

    save_path = "/var/tmp/mi714/NEW/npy_dataset"
    turn_npy(path, save_path)
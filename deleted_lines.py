def resize(image, input_folder, size):
    """Defining a function that will allow me to parallelise the resizing process.
    It takes the name (basename) of the current image, resizes and saves the image.

    Args:
        input_path (string): Path to the image.
        size (tuple): Target size to be resized to.
    """
    image_path = input_folder + "/" + image + ".png"
    img = Image.open(image_path)
    img = img.resize((size[0], size[1]), resample=Image.ANTIALIAS)
    img.save(image_path)


def resize_set(input_folder, size, jobs, train_val, img_mask):
    """
    Stores the input and output directories, then stores all the
    names of the images in a list, and executes the resizing in parallel.
    For the parallelisation, Parallel and delayed are used.
    tqdm is used for visual representation of the progress, since the
    dataset is around 30GB, it will take some time to process.

    Args:
        input_folder (string): Path for input folder.
        size (tuple): Target size to be resized to.
        jobs (int): Number of parallelised jobs.
        train_val (string): Specifies whether it's "Train" or "Validation"; used for showing progress.
        is_mask (string): States whether it's an "Image" or "Mask" set; used for showing progress.
    """
    images = [splitext(file)[0] for file in listdir(input_folder)]
    print(f"Resizing {train_val} {img_mask}.")
    Parallel(n_jobs=jobs)(
        delayed(resize)(image, input_folder, size)
        for image in tqdm(images)
    )





    # Resize images.
    resize_set(
        images_folder_path,
        resize_dimensions,
        n_jobs,
        "Train",
        "Images",
    )

    # Resize masks.
    resize_set(
        masks_folder_path,
        resize_dimensions,
        n_jobs,
        "Train",
        "Masks",
    )
#############################################################################

    # Make images greyscale.
    make_greyscale(
        images_folder_path,
        n_jobs,
    )

def turn_grayscale(image, folder_path):
    """Function for parallelising the grayscale process.

    Args:
        image (string): ID of image to be turn into grayscale.
        folder_path (string): Path leading to folder containing images.
    """
    img = Image.open(folder_path + "/" + image + ".png")
    grey = transforms.functional.rgb_to_grayscale(img)
    grey.save(folder_path + "/" + image + ".png")


def make_greyscale(folder_path, jobs):
    """Turns all images in a folder from RGB to grayscale.

    Args:
        folder_path (string): Path leading to folder containing images.
        jobs (int): Number of job for parallelisation.
    """
    images = [splitext(file)[0] for file in listdir(folder_path)]
    print("Turning images to GrayScale:")
    Parallel(n_jobs=jobs)(
        delayed(turn_grayscale)(image, folder_path) for image in tqdm(images)
    )
    logging.info(f"Successfully turned {len(images)} images to GrayScale.")

#############################################################################





        if melanoma == 0:
            augm_probability = 0.5
            n = random.random()
            if n < augm_probability:
                # Perform augmentation, store the resulting image and mask.
                img_1, img_1_mask = augment_operations(
                    image_id, images_folder_path, masks_folder_path, train_val
                )

                # Save image and mask in two dedicated folders.
                img_1.save(images_folder_path + "/" + image_id + "x1" + ".png", "PNG", quality=100)
                img_1_mask.save(
                    masks_folder_path + "/" + image_id + "_segmentation" + "x1" + ".png", "PNG", quality=100
                )

                # Add new datapoint to .csv file 
                with open(csv_file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([image_id + "x1", 0, 0])

#############################################################################

def divide_imgs_by_class(path):
    """Splits images in "melanoma" or "no_melanoma" folders.
    Uses "get_result()" function to retrieve diagnosis result.

    Args:
        path (string): Path to folder containing dataset.
    """    
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

#############################################################################
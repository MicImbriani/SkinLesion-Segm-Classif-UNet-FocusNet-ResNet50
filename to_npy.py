from os import listdir
from os.path import splitext
import cv2
from PIL import Image

def turn_npy_imgs(folder_path):    
    images = [splitext(file)[0] for file in listdir(folder_path)]
    imgs_array = []
    for image in images:
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
    for image in images:
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

def train_val_split(path):
    images_folder_path = path + "/" + "Train"
    masks_folder_path = images_folder_path + "_GT_masks"
    val_imgs_folder_path = path + "/" + "Validation"
    val_masks_folder_path = val_imgs_folder_path + "_GT_masks"
    test_imgs_folder_path = path + "/" + "Test"
    test_masks_folder_path = val_imgs_folder_path + "_GT_masks"

    train_X = turn_npy_imgs(images_folder_path)
    train_y, train_y1 = turn_npy_masks(masks_folder_path)

    val_X = turn_npy_imgs(val_imgs_folder_path)
    val_y, val_y1 = turn_npy_masks(val_masks_folder_path)

    test_X = turn_npy_imgs(test_imgs_folder_path)
    test_y, test_y1 = turn_npy_masks(test_masks_folder_path)


    train_X = train_X.astype('float32')
    val_X = val_X.astype('float32')
    test_X = test_X.astype('float32')

    mean = np.mean(train_X)  # mean for data centering
    std = np.std(train_X)  # std for data normalization

    train_X -= mean
    train_X /= std

    val_X -= mean 
    val_X /= std

    test_X /= std
    test_X -= mean 

    trainMask = trainMask.astype('float32')
    trainMask /= 255.  # scale masks to [0, 1]

    valMask = valMask.astype('float32')
    valMask /= 255.  # scale masks to [0, 1]

    testMask = testMask.astype('float32')
    testMask /= 255.  # scale masks to [0, 1]


    np.save('/var/tmp/mi714/test_new_npy2/data.npy',train_X)
    np.save('/var/tmp/mi714/test_new_npy2/dataMask.npy',train_y)
    np.save('/var/tmp/mi714/test_new_npy2/dataval.npy', val_X)
    np.save('/var/tmp/mi714/test_new_npy2/dataMaskval.npy', val_y)
    np.save('/var/tmp/mi714/test_new_npy2/datatest.npy', test_X)
    np.save('/var/tmp/mi714/test_new_npy2/dataMasktest.npy', test_y)

    
path = "/var/tmp/mi714/aug17"
train_val_split(path)
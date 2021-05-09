import os, glob
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from os.path import splitext
from os import listdir
from PIL import Image
from networks.focusnet_nn import focusnet
from networks.unet_nn import unet
from networks.unet_res_se import get_unet




size = (256,256,1)
#unet = unet(size, batch_norm=False)
unetresse = get_unet()
focus = focusnet()

#unet.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/unet_nobn/unet_nobn1/unet_nobn_weights.h5")
#unetbn.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/unet_bn/unet_bn/unet_bn_weights.h5")
unetresse.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/unet_polished/unet_polished_weights.h5")
#focus.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/focusnet99/focus_weights.h5")



# trainData = np.load('/var/tmp/mi714/test_new_npy2/data.npy')
# trainMask = np.load('/var/tmp/mi714/test_new_npy2/dataMask.npy')

# valData = np.load('/var/tmp/mi714/test_new_npy2/dataval.npy')
# valMask = np.load('/var/tmp/mi714/test_new_npy2/dataMaskval.npy')



#p = focus.predict(trainData)


# for i in range(p.shape[0]):
#     img = p[i]
#     #print(img[0][61][0])
#     ret, img = cv2.threshold(img, 0.5, 255, cv2.THRESH_BINARY)
#     # print(img.shape)
#     # print()
#     # print()
#     img = img.astype(np.uint8)
#     #img = np.squeeze(img, axis=2)  # axis=2 is channel dimension 
#     img = Image.fromarray(img)
#     newImage = img.convert("L")
#     newImage.save("/var/tmp/mi714/test_new_npy2/unet_bn_preds/" + str(i) + ".png")





# path is the path with actual images
def generate_new(model, path):
    images = [file for file in listdir(path)]
    #files = os.listdir(path)
    for image in images:
        img_id = splitext(image)[0]
        x = Image.open(path + "/" + image)
        x = np.asarray(x, dtype=np.float32)
        x = x[...,np.newaxis]
        x = x[np.newaxis,...]
        x = np.asarray(x)
        print(x.shape)
        img = model.predict(x)
        ret, img = cv2.threshold(img, 0.5, 255, cv2.THRESH_BINARY)
        img = img.astype(np.float64)
        img = Image.fromarray(img)
        img = img.convert("L")
        img.save(save_path + "/" + image)


dataset_path = "/var/tmp/mi714/aug17/Validation"
save_path = "/var/tmp/mi714/test_new_npy2/predictions/unetseres/val"
generate_new(focus, dataset_path)

dataset_path = "/var/tmp/mi714/aug17/Train"
save_path = "/var/tmp/mi714/test_new_npy2/predictions/unetseres/train"
generate_new(focus, dataset_path)

dataset_path = "/var/tmp/mi714/aug17/Test"
save_path = "/var/tmp/mi714/test_new_npy2/predictions/unetseres/test"
generate_new(focus, dataset_path)
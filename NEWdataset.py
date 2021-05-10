import os, glob
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from os.path import splitext
from os import listdir
from PIL import Image
from networks.focusnet_nn import focusnet
from networks.unet_nn import unet
from networks.unet_res_se_nn import unet_res_se
from networks.focus import get_focusnetAlpha




size = (256,256,1)
#unet = unet(size, batch_norm=False)
# unetresse = get_unet()
# focus = focusnet()
model = get_focusnetAlpha()

#unet.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/unet_nobn/unet_nobn1/unet_nobn_weights.h5")
#unetbn.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/unet_bn/unet_bn/unet_bn_weights.h5")
#unetresse.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/unet_polished/unet_polished_weights.h5")
#focus.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/focusnet99/focus_weights.h5")
model.load_weights("/var/tmp/mi714/NEW/models/focusnet/focusnet_weights.h5")



# trainData = np.load('/var/tmp/mi714/NEW/npy_dataset/data.npy')
# trainMask = np.load('/var/tmp/mi714/NEW/npy_dataset/dataMask.npy')

valData = np.load('/var/tmp/mi714/NEW/npy_dataset/dataval.npy')
valMask = np.load('/var/tmp/mi714/NEW/npy_dataset/dataMaskval.npy')



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
        x = Image.open(path + "/" + image).convert('L')
        x = np.asarray(x, dtype=np.float32)
        x = x[np.newaxis,...]   
        x = x[...,np.newaxis]
        x = np.asarray(x)
        img = model.predict(x)
        print(img.shape)
        # ret, img = cv2.threshold(img, 0.5, 255, cv2.THRESH_BINARY)
        # print(img.shape)
        img = np.asarray(img, dtype=np.float32)
        img = img[0,:,:,0]
        print(img.shape)
        img = Image.fromarray(img)
        img = img.convert("L")
        print("#########")
        img.save(save_path + "/" + image)


dataset_path = "/var/tmp/mi714/NEW/aug_dataset/ISIC-2017_Validation_Data"
save_path = "/var/tmp/mi714/NEW/predictions/focusnet/val"
generate_new(model, dataset_path)
import numpy as np

from keras.optimizers import Adam, SGD

import metrics
from networks.focusnet_nn import focusnet
from networks.unet_nn import unet
from networks.unet_res_se_nn import unet_res_se
from networks.focus import get_focusnetAlpha



# #U-Net
# model = unet(batch_norm=False)
# model.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/unet_nobn/unet_nobn1/unet_nobn_weights.h5")
# #U-Net BatchNorm
# model = unet(batch_norm=True)
# #model.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/unet_bn/unet_bn/unet_bn_weights.h5")
# #U-Net Res SE
# model = unet_res_se()
# #model.load_weights("/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/unet_polished/unet_polished_weights.h5")
#Focusnet
model = get_focusnetAlpha()
model.load_weights("/var/tmp/mi714/NEW/models/focusnet/focusnet_weights.h5")


# trainData = np.load('/var/tmp/mi714/test_new_npy2/data.npy')
# trainMask = np.load('/var/tmp/mi714/test_new_npy2/dataMask.npy')

# valData = np.load('/var/tmp/mi714/test_new_npy2/dataval.npy')
# valMask = np.load('/var/tmp/mi714/test_new_npy2/dataMaskval.npy')

# testData = np.load('/var/tmp/mi714/test_new_npy2/datatest.npy')
# testMask = np.load('/var/tmp/mi714/test_new_npy2/dataMasktest.npy')

valData = np.load('/var/tmp/mi714/NEW/npy_dataset/dataval.npy')
valMask = np.load('/var/tmp/mi714/NEW/npy_dataset/dataMaskval.npy')


# my_adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
# model.compile(optimizer=my_adam,
#                 loss=metrics.dice_coef_loss,
#                 metrics=[metrics.dice_coef_loss,
#                         metrics.jaccard_coef_loss,
#                         metrics.true_positive,
#                         metrics.true_negative,
#                         ])

# score = model.evaluate(valData, valMask, verbose=1)
# dice_coef_loss = score[0]
# jac_indx_loss = score[1]
# true_positive = score[2]
# true_negative = score[3]
# acc = score[4]

# print(f"""
# RESULTS: 
# Dice Coefficient Loss: {dice_coef_loss}
# Jaccard Index Loss: {jac_indx_loss}
# True Positive: {true_positive}
# True Negative: {true_negative}
# """)
##########################################################
save_path = "/var/tmp/mi714/NEW/predictions/focusnet"
from PIL import Image
oi = valData[143]
img = np.asarray(oi, dtype=np.float32)
print(img.shape)
print(img[127][123][0])
#img = img[:,:,0]
print(img.shape)
img = Image.fromarray(img)
img = img.convert("L")
print("#########")
#img.save(save_path + "/asd.png")


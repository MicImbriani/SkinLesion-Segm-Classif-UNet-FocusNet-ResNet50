import numpy as np

from keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import AUC

import metrics
from networks.unet_nn import unet
from networks.unet_res_se_nn import unet_res_se
from networks.focus import get_focusnetAlpha
from networks.resnet import get_res
from data_processing.generate_new_dataset import generate_targets
from tensorflow.keras.applications.resnet50 import preprocess_input




########### SEGMENTATION ###########
# U-Net
model = unet(batch_norm=False)
model.load_weights("/var/tmp/mi714/NEW/models/UNET/unet10/unet10_weights.h5")
# U-Net BatchNorm
# model = unet(batch_norm=True)
# model.load_weights("/var/tmp/mi714/NEW/models/UNET_BN/unet_bn10/unet_bn10_weights.h5")
# U-Net Res SE
# model = unet_res_se()
# model.load_weights("/var/tmp/mi714/NEW/models/UNET_RES_SE/unet_res_se10/unet_res_se10_weights.h5")
#Focusnet
# model = get_focusnetAlpha()
# model.load_weights("/var/tmp/mi714/NEW/models/FOCUS/focusnet10/focusnet10_weights.h5")

########### CLASSIFICATION ###########
# model = get_res()
# Original
# model.load_weights("/var/tmp/mi714/NEW/models/RESNETS/RESNET_OG/resnet_og/resnet_weights.h5")
# U-Net
# model.load_weights()
# U-Net BatchNorm
# model.load_weights()
# Res SE U-Net
# model.load_weights()
# FocusNet
# model.load_weights()




# Data, Masks & Classification target labels
# trainData = np.load('/var/tmp/mi714/test_new_npy2/data.npy')
# valData = np.load('/var/tmp/mi714/test_new_npy2/dataval.npy')
testData = np.load('/var/tmp/mi714/NEW/npy_dataset/datatest.npy')


# Segmentation masks
# trainMask = np.load('/var/tmp/mi714/test_new_npy2/dataMask.npy')
# valMask = np.load('/var/tmp/mi714/test_new_npy2/dataMaskval.npy')
testMask = np.load('/var/tmp/mi714/NEW/npy_dataset/dataMasktest.npy')





########### SEGMENTATION ###########

X = testData
y = testMask

X = X.astype('float32')
y /= 255.  # scale masks to [0, 1]


my_adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(optimizer=my_adam,
                loss=metrics.focal_loss,
                metrics=[metrics.dice_coef_loss,
                        metrics.jaccard_coef_loss,
                        metrics.true_positive,
                        metrics.true_negative,
                        ])

score = model.evaluate(X, y, verbose=1)
dice_coef_loss = score[1]
jac_indx_loss = score[2]
true_positive = score[3]
true_negative = score[4]


print(f"""
RESULTS: 
Dice Coefficient Loss: {dice_coef_loss}
Jaccard Index Loss: {jac_indx_loss}
True Positive: {true_positive}
True Negative: {true_negative}
""")





########### CLASSIFICATION ###########

# # Classification data
# # x_train = np.concatenate((trainData,)*3, axis=-1)
# # x_train = preprocess_input(x_train)

# # x_val = np.concatenate((valData,)*3, axis=-1)
# # x_val = preprocess_input(x_val)

# x_test = np.concatenate((testData,)*3, axis=-1)
# x_test = preprocess_input(x_test)

# # Classification target labels 
# path = "/var/tmp/mi714/NEW/aug_dataset/"

# # y_train = generate_targets(path + "ISIC-2017_Training_Data",
# #                            path + "ISIC-2017_Training_Part3_GroundTruth.csv")

# # y_val = generate_targets(path + "ISIC-2017_Validation_Data",
# #                          path + "ISIC-2017_Validation_Part3_GroundTruth.csv")

# y_test = generate_targets(path + "ISIC-2017_Test_v2_Data",
#                           path + "ISIC-2017_Test_v2_Part3_GroundTruth.csv")

# X = x_test
# y = y_test

# my_adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

# # Compile model and print summary
# rocauc = AUC(num_thresholds=200,
#             curve="ROC",
#             summation_method="interpolation",
#             name=None,
#             dtype=None,
#             thresholds=None,
#             multi_label=False,
#             label_weights=None,
#             )

# model.compile(loss='categorical_crossentropy',
#               optimizer=my_adam,
#               metrics=[metrics.sensitivity,
#                        metrics.specificity,
#                        rocauc,
#                        'acc'
#                        ])

# score = model.evaluate(X, y, verbose=1)
# binary_ce = score[0]
# sensitivity = score[1]
# specificity = score[2]
# rocauc = score[3]
# acc = score[4]

# print(f"""
# RESULTS: 
# Binary Cross-Entropy Loss: {binary_ce}
# Sensitivity: {sensitivity}
# Specificity: {specificity}
# AUC ROC: {rocauc}
# Accuracy: {acc}
# """)


import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# from networks.unet_nn import unet
from networks.unet_res_se_nn import unet_res_se

# from networks.focus import get_focusnetAlpha

import metrics
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from keras.optimizers import Adam, SGD
from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)
from keras.metrics import MeanIoU


trainData = np.load("/var/tmp/mi714/NEW/npy_dataset/data.npy")
trainMask = np.load("/var/tmp/mi714/NEW/npy_dataset/dataMask.npy")

valData = np.load("/var/tmp/mi714/NEW/npy_dataset/dataval.npy")
valMask = np.load("/var/tmp/mi714/NEW/npy_dataset/dataMaskval.npy")


trainMask = trainMask.astype("float32")
trainMask /= 255.0  # scale masks to [0, 1]

valMask = valMask.astype("float32")
valMask /= 255.0  # scale masks to [0, 1]


model_name = "unet_res_se"

path = "/var/tmp/mi714/NEW/models/unet_res_se/"
os.makedirs(path, exist_ok=True)

# Selection of which model to train
# model = unet(batch_norm=False)
# model = unet(batch_norm=True)
model = unet_res_se()
# model = get_focusnetAlpha()

my_adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
my_sgd = SGD(lr=0.00001, momentum=0.9, decay=1e-6, nesterov=True)

model.compile(
    optimizer=my_adam,
    loss=metrics.dice_coef_loss,
    metrics=[
        metrics.dice_coef_loss,
        metrics.jaccard_coef_loss,
        metrics.true_positive,
        metrics.true_negative,
    ],
)
model.summary()

callbacks = [
    ModelCheckpoint(
        path + "/" + model_name + "_weights.h5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
    ),
    EarlyStopping(
        patience=10, monitor="val_loss", mode="min", verbose=1, min_delta=0.001
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        min_delta=0.01,
        cooldown=0,
        min_lr=1e-7,
        factor=0.2,
        patience=5,
        verbose=1,
    ),
]

history = model.fit(
    trainData,
    trainMask,
    batch_size=6,
    epochs=100,
    verbose=1,
    validation_data=(valData, valMask),
    shuffle=True,
    callbacks=callbacks,
)

model.save(path + "/" + model_name + "_model.h5")

########################################################################################################################################################################


# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(path + "/loss.png")
plt.clf()
# summarize history for jaccard
plt.plot(history.history["jaccard_coef_loss"])
plt.plot(history.history["val_jaccard_coef_loss"])
plt.title("model jaccard coef loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(path + "/jaccard.png")
plt.clf()
# summarize history for positive
plt.plot(history.history["true_positive"])
plt.plot(history.history["val_true_positive"])
plt.title("model true positive")
plt.ylabel("positives")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(path + "/pos.png")
plt.clf()
# summarize history for negative
plt.plot(history.history["true_negative"])
plt.plot(history.history["val_true_negative"])
plt.title("model true negative")
plt.ylabel("negatives")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(path + "/neg.png")

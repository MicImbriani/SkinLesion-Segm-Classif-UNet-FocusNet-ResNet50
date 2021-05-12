from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from metrics import (
    dice_coef_loss,
    auroc,
    jaccard_coef_loss,
    true_positive,
    true_negative,
    sensitivity,
    specificity,
)
from tensorflow.keras.metrics import AUC
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt


trainData = np.load("/var/tmp/mi714/test_new_npy2/data.npy")
trainMask = np.load("/var/tmp/mi714/test_new_npy2/dataMask.npy")

valData = np.load("/var/tmp/mi714/test_new_npy2/dataval.npy")
valMask = np.load("/var/tmp/mi714/test_new_npy2/dataMaskval.npy")


trainData = np.stack((trainData,) * 3, axis=-1)
print(trainData.shape)

valData = np.stack((valData,) * 3, axis=-1)
print(valData.shape)


trainData = trainData.astype("float32")
mean = np.mean(trainData)  # mean for data centering
std = np.std(trainData)  # std for data normalization

valData = valData.astype("float32")

trainData -= mean
trainData /= std

valData -= mean
valData /= std

trainMask = trainMask.astype("float32")
trainMask /= 255.0  # scale masks to [0, 1]

valMask = valMask.astype("float32")

########################################################################################################################################################################valMask /= 255.  # scale masks to [0, 1]
from keras.models import Sequential
from keras.layers import Dense, Input


def get_res():

    model = Sequential()
    # new_input = Input(shape=(256, 256, 1))
    model.add(ResNet50(include_top=False, pooling="avg", weights=None))
    # input_shape=(256,256,3),
    # input_tensor=new_input,
    # classes=2))

    # 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
    model.add(Dense(2, activation="softmax"))

    # Say not to train first layer (ResNet) model as it is already trained
    model.layers[0].trainable = False

    return model


from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


# preprocessing_function is applied on each image but only after re-sizing & augmentation (resize => augment => pre-process)
# Each of the keras.application.resnet* preprocess_input MOSTLY mean BATCH NORMALIZATION (applied on each batch) stabilize the inputs to nonlinear activation functions
# Batch Normalization helps in faster convergence
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

path = "/var/tmp/mi714/class_division"
train_path = path + "/Train"
val_path = path + "/Validation"

image_size = 256
bs = 8
# flow_From_directory generates batches of augmented data (where augmentation can be color conversion, etc)
# Both train & valid folders must have NUM_CLASSES sub-folders
train_generator = data_generator.flow_from_directory(
    train_path,
    target_size=(image_size, image_size),
    batch_size=bs,
    class_mode="categorical",
)

validation_generator = data_generator.flow_from_directory(
    val_path,
    target_size=(image_size, image_size),
    batch_size=bs,
    class_mode="categorical",
)

########################################################################################################################################################################

print((bs, len(train_generator), bs, len(validation_generator)))

path = "/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/resnet/resnet_base(del)/"

try:
    os.makedirs(
        "/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/resnet/renets_base(del)",
        exist_ok=True,
    )
except:
    pass


checkpoint = ModelCheckpoint(
    "resnet_base_weights.h5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="min",
)
early_stopping = EarlyStopping(
    patience=10, monitor="val_loss", mode="min", verbose=1, min_delta=0.01
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    min_delta=0.01,
    cooldown=0,
    min_lr=0.5e-7,
    factor=0.5,
    patience=5,
    verbose=1,
)


model = get_res()
from metrics import ROCAUC

rocauc = tf.keras.metrics.AUC(
    num_thresholds=200,
    curve="ROC",
    summation_method="interpolation",
    name=None,
    dtype=None,
    thresholds=None,
    multi_label=False,
    label_weights=None,
)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
my_adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(
    loss="categorical_crossentropy",
    optimizer=my_adam,
    metrics=[sensitivity, specificity, rocauc, "acc"],
)


history = model.fit_generator(
    train_generator,
    # steps_per_epoch=10,
    epochs=100,
    validation_data=validation_generator,
    # validation_steps=10,
    callbacks=[checkpoint, reduce_lr, early_stopping],
)
# model.load_weights("../working/best.hdf5")

# history = model.fit(trainData,
#                     trainMask,
#                     epochs=1,
#                     batch_size = 6,
#                     validation_data=(valData, valMask),
#                     verbose=2)

model.save(path + "resnet_base_model.h5")

## TAKEN FROM
# https://github.com/bnsreenu/python_for_microscopists/blob/master/203b_skin_cancer_lesion_classification_V4.0.py
# # Prediction on test data
# y_pred = model.predict(x_test)
# # Convert predictions classes to one hot vectors
# y_pred_classes = np.argmax(y_pred, axis = 1)
# # Convert test data to one hot vectors
# y_true = np.argmax(y_test, axis = 1)

# #Print confusion matrix
# cm = confusion_matrix(y_true, y_pred_classes)

# fig, ax = plt.subplots(figsize=(6,6))
# sns.set(font_scale=1.6)
# sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)


# #PLot fractional incorrect misclassifications
# incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
# plt.bar(np.arange(7), incorr_fraction)
# plt.xlabel('True Label')
# plt.ylabel('Fraction of incorrect predictions')


# summarize history for accuracy
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(path + "accuracy.png")
plt.clf()
# summarize history for AUC
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model AUC")
plt.ylabel("AUC")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(path + "AUC.png")
plt.clf()
# summarize history for sensitivity
plt.plot(history.history["sensitivity"])
plt.plot(history.history["val_sensitivity"])
plt.title("model sensitivity")
plt.ylabel("sensitivity")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(path + "sensitivity.png")
plt.clf()
# summarize history for specificity
plt.plot(history.history["specificity"])
plt.plot(history.history["val_specificity"])
plt.title("model specificity")
plt.ylabel("specificity")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.savefig(path + "specificity.png")
plt.clf()


# preds = model.predict(x)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])
# # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

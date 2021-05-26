import os 
import numpy as np
import matplotlib.pyplot as plt

import keras
from tensorflow.keras.metrics import AUC
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from metrics import (
    true_positive,
    true_negative,
    sensitivity,
    specificity)
from networks.resnet import get_res

from tensorflow.keras.utils import to_categorical
from data_processing.data_process import get_result



def generate_targets(path, csv_path):
        images = os.listdir(path)
        images.sort()
        labels = []

        for image in images:
                image_id = os.path.splitext(image)[0]
                temp = get_result(image_id, csv_path)
                labels.append(temp)
        labels = np.array(labels)
        labels = labels[:, np.newaxis]

        y_array = to_categorical(labels, 2)
        y_array = np.array(y_array)
        return y_array 
        
# Generate target array for train set
train_path = "/var/tmp/mi714/NEW/ISIC-2017_Training_Data"
train_csv_path = "/var/tmp/mi714/NEW/ISIC-2017_Training_Part3_GroundTruth.csv"
# path = "/var/tmp/mi714/NEW/aug_dataset/ISIC-2017_Training_Data"
# csv_path = "/var/tmp/mi714/NEW/aug_dataset/ISIC-2017_Training_Part3_GroundTruth.csv"
y_train = generate_targets(train_path, train_csv_path)
print(y_train.shape)

# Generate target array for validation set
val_path = "/var/tmp/mi714/NEW/ISIC-2017_Validation_Data"
val_csv_path = "/var/tmp/mi714/NEW/ISIC-2017_Validation_Part3_GroundTruth.csv"
# path = "/var/tmp/mi714/NEW/aug_dataset/ISIC-2017_Validation_Data"
# csv_path = "/var/tmp/mi714/NEW/aug_dataset/ISIC-2017_Validation_Part3_GroundTruth.csv"
y_val = generate_targets(val_path, val_csv_path)
print(y_val.shape)

# Load train data and "convert" to RGB by expanding the channel dimension
x_train = np.load('/var/tmp/mi714/NEW/data.npy')
# x_train = np.load('/var/tmp/mi714/NEW/npy_dataset/data.npy')
x_train = np.concatenate((x_train,)*3, axis=-1)
x_train = preprocess_input(x_train)
print(x_train.shape)

# Load validation data and "convert" to RGB by expanding the channel dimension
x_val = np.load('/var/tmp/mi714/NEW/dataval.npy')
# x_val = np.load('/var/tmp/mi714/NEW/npy_dataset/dataval.npy')
x_val = np.concatenate((x_val,)*3, axis=-1)
x_val = preprocess_input(x_val)
print(x_val.shape)

# Rescale the pixel values for data normalisation
x_train /= 255
x_val /= 255





model_name = "resnet"

path = "/var/tmp/mi714/NEW/models/RESNETS/RESNET_OG/" + model_name
os.makedirs(path, exist_ok=True)

model = get_res()


rocauc = AUC(num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            name=None,
            dtype=None,
            thresholds=None,
            multi_label=False,
            label_weights=None,
            )


sgd = SGD(lr = 0.00001, decay = 1e-6, momentum = 0.9, nesterov = True)
my_adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

model.compile(loss='categorical_crossentropy',
              optimizer=my_adam,
              metrics=[sensitivity,
                       specificity,
                       rocauc,
                       'acc'
                       ])

model.summary()

checkpoint = ModelCheckpoint(path + "/" + model_name + "_weights.h5", 
                             monitor='val_auc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='max'
                             )
early_stopping = EarlyStopping(patience=10,
                               monitor='val_auc',
                               mode='max',
                               verbose=1,
                               min_delta=0.01
                               )
reduce_lr = ReduceLROnPlateau(monitor='val_auc',
                              mode='max',
                              min_delta=0.01,
                              cooldown=0,
                              min_lr=0.5e-7,
                              factor=0.5,
                              patience=5,
                              verbose=1
                              )

history = model.fit(x_train,
                y_train,
                batch_size = 16 ,
                # steps_per_epoch=10,
                epochs=50,
                validation_data=(x_val,y_val),
                # validation_steps=10,
                callbacks=[checkpoint,
                        reduce_lr, 
                        early_stopping
                        ],
                )

model.save(path + "/" + model_name + "_model.h5")



# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(path + 'accuracy.png')
plt.clf()
# summarize history for AUC
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model AUC')
plt.ylabel('AUC')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(path + 'AUC.png')
plt.clf()
# summarize history for sensitivity
plt.plot(history.history['sensitivity'])
plt.plot(history.history['val_sensitivity'])
plt.title('model sensitivity')
plt.ylabel('sensitivity')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(path + 'sensitivity.png')
plt.clf()
# summarize history for specificity
plt.plot(history.history['specificity'])
plt.plot(history.history['val_specificity'])
plt.title('model specificity')
plt.ylabel('specificity')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(path + 'specificity.png')
plt.clf()

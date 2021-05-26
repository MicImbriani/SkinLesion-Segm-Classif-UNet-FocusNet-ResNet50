import os 
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.metrics import AUC
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator

from metrics import (
    true_positive,
    true_negative,
    sensitivity,
    specificity)
from networks.resnet import get_res

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD





# Folders with images divided by diagnosis result
path = "/var/tmp/mi714/NEW/classif_dataset"
train_path = path + "/ISIC-2017_Training_Data"
val_path = path + "/ISIC-2017_Validation_Data"

image_size = 256
bs = 8



# Preprocessing_function is applied on each image
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# flow_From_directory generates batches of augmented data
train_generator = data_generator.flow_from_directory(
        train_path,
        target_size=(image_size, image_size),
        batch_size=bs,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        val_path,
        target_size=(image_size, image_size),
        batch_size=bs,
        class_mode='categorical') 



print(len(train_generator), len(validation_generator))



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
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min'
                             )
early_stopping = EarlyStopping(patience=10,
                               monitor='val_loss',
                               mode='min',
                               verbose=1,
                               min_delta=0.01
                               )
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              mode='min',
                              min_delta=0.01,
                              cooldown=0,
                              min_lr=0.5e-7,
                              factor=0.5,
                              patience=5,
                              verbose=1
                              )

history = model.fit_generator(train_generator,
                              # steps_per_epoch=10,
                              epochs=50,
                              validation_data=validation_generator,
                              # validation_steps=10,
                              callbacks=[checkpoint,
                                        reduce_lr, 
                                        early_stopping
                                        ],
                                )

model.save(path + "/" + model_name + "_model.h5")

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


# preds = model.predict(x)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])
# # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
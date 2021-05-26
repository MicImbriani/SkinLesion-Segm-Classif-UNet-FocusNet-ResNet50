import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import VGG16

def get_res():
    # new_input = Input(256, 256, 3)

    res = ResNet50(include_top = False, 
                    #pooling = 'avg',
                    weights = "imagenet",
                    input_shape=(256,256,3))
                    # input_tensor=new_input)
    # for layer in res.layers[:-15]:
    #         layer.trainable = False

    # for i, layer in enumerate(res.layers):
    #         print(i, layer.name, "-", layer.trainable)

    res_out = res.output
    
    gap = GlobalAveragePooling2D()(res_out)
    dropout = Dropout(0.35)(gap)
    # flatten = Flatten()(dropout)
    # model.add(keras.layers.Flatten())
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(128, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(64, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.BatchNormalization())
    # 2nd layer as Dense for 2-class classification
    out = Dense(2, activation = 'sigmoid')(dropout)

    # Say not to train first layer (ResNet) model as it is already trained
    # model.layers[0].trainable = False
    
    model = Model(inputs = res.input, outputs = out)

    return model

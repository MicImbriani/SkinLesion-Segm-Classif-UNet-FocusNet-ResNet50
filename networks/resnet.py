import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import VGG16

def get_res():
    res = ResNet50(include_top = False, 
                    #pooling = 'avg',
                    weights = "imagenet",
                    input_shape=(256,256,3))
    res_out = res.output
    gap = GlobalAveragePooling2D()(res_out)
    dropout = Dropout(0.35)(gap)
    out = Dense(2, activation = 'sigmoid')(dropout)
    
    model = Model(inputs = res.input, outputs = out)

    return model

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:23:01 2019
@author: chaitanya
"""

from keras.layers import Conv1D, Conv2D, Activation, Multiply, Add, Concatenate, BatchNormalization, add
#from Batch_Normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from se import squeeze_excite_block

def initial_conv_block(weight_decay=5e-4):
    ''' Adds an initial convolution block, with batch normalization and relu activation
    Args:
        input: input tensor
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    def f(input):
        x = Conv2D(48, 3, padding='same', use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(input)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU()(x)
        return x 
    return f



#def _bn_Lrelu(input):
#    """Helper to build a BN -> relu block
#    """
#    norm = BatchNormalization(axis=3, freeze=False)(input)
#    return LeakyReLU()(norm)
#
#def _conv_bn_Lrelu(**conv_params):
#    """Helper to build a conv -> BN -> relu block
#    """
#    filters = conv_params["filters"]
#    kernel_size = conv_params["kernel_size"]
#    strides = conv_params.setdefault("strides", (1, 1))
#    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
#    padding = conv_params.setdefault("padding", "same")
#    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))
#
#    def f(input):
#        conv = Conv2D(filters=filters, kernel_size=kernel_size,
#                      strides=strides, padding=padding,
#                      kernel_initializer=kernel_initializer,
#                      kernel_regularizer=kernel_regularizer)(input)
#        return _bn_Lrelu(conv)
#
#    return f
    

def basic_2d(filters, block=0, stride=None):
    if stride is None:
        if block != 0:
            stride = 1
        else:
            stride = 2


    def f(input):
        y = Conv2D(filters, 3, strides=stride, padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(input)
        y = BatchNormalization(axis=3)(y)
        y = LeakyReLU()(y)
        y = Conv2D(filters, 3, strides=1, padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(y)
        y = BatchNormalization(axis=3)(y)
        if block == 0:
            shortcut = Conv2D(filters, 1, strides=stride, padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(1e-4))(input)
            shortcut = BatchNormalization(axis=3)(shortcut)
        else:
            shortcut = input
        y = Add()([y, shortcut])
        y = LeakyReLU()(y)
        return y
    return f

def branch(filters):#, dilationrate1D, dilationrate2D):
    
    def f(input):
        b = Conv1D(filters=filters, kernel_size=1,
                   strides=1, padding="same",
                   #dilation_rate=dilationrate1D,
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2(1e-4))(input)
        b = BatchNormalization(axis=3)(b)
        b = LeakyReLU()(b)
        b_res = b
    
        b = Conv1D(filters=filters, kernel_size=1,
                   strides=1, padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2(1e-4))(b)
        b = BatchNormalization(axis=3)(b)
    
        b_attention = Activation("softmax")(b)
        
        b = Conv2D(filters=filters, kernel_size=3,
                   strides=(1,1), padding="same",
                   #dilation_rate=dilationrate2D,
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2(1e-4))(b)
        b = BatchNormalization(axis=3)(b)
        b = Multiply()([b, b_attention])
        b = LeakyReLU()(b)
    
        b = Conv1D(filters=filters, kernel_size=1,
                   strides=1, padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2(1e-4))(b)
        b = BatchNormalization(axis=3)(b)
        b = Add()([b, b_res])
        b = LeakyReLU()(b)
    
        return b
    return f

def focusnetAlphaLayer(filters):
    
    def f(input):
        print("#############")
        print(input.shape)
        x = Conv2D(filters=filters, kernel_size=3,
                   strides=1, padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2(1e-4))(input)
        x = BatchNormalization(axis=3)(x)
        x = LeakyReLU()(x)
        print(x.shape)
        
        #x1 = branch(filters//3)(x)#, dilationrate1D=1, dilationrate2D=(1,1))(x)
        #x2 = branch(filters//3)(x)#, dilationrate1D=2, dilationrate2D=(2,2))(x)
        #x3 = branch(filters//3)(x)#, dilationrate1D=3, dilationrate2D=(3,3))(x)
        
        #x_concatenated = Concatenate()([x1,x2,x3], axis=3)
        
        x = Conv2D(filters=filters, kernel_size=3,
                   strides=(1,1), padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=l2(1e-4))(x)#(x_concatenated)
        x = BatchNormalization(axis=3)(x)
        x = squeeze_excite_block(x)
        x_res = input
        x_res = Conv2D(filters=filters, kernel_size=1, strides=1, padding="same", kernel_initializer="he_normal")(x_res)
        print(x.shape)
        print(x_res.shape)
        x = Add()([x, x_res])
        
        x =  LeakyReLU()(x)
        
        return x
    return f
    
########################################################################################################################################################################
    
import numpy as np
from keras.models import Model
from keras.layers import Input, UpSampling2D
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K


def get_focusnetAlpha():
    
    inputs = Input((256, 256, 1))
    
    conv1 = initial_conv_block()(inputs)

    conv1 = focusnetAlphaLayer(84)(conv1)
    pool1 = basic_2d(84, block=0)(conv1)
    
    conv2 = focusnetAlphaLayer(144)(pool1)
    pool2 = basic_2d(144, block=0)(conv2)
    
    conv3 = focusnetAlphaLayer(255)(pool2)
    pool3 = basic_2d(255, block=0)(conv3)
    
    conv4 = focusnetAlphaLayer(396)(pool3)
    pool4 = basic_2d(396, block=0)(conv4)
    
    
    bottleneck = focusnetAlphaLayer(510)(pool4)
    
    up1 = Conv2D(396, 2, activation = LeakyReLU(), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(bottleneck))
    merge1 = Add()([conv4, up1])
    conv5 = focusnetAlphaLayer(396)(merge1)
    
    up2 = Conv2D(255, 2, activation = LeakyReLU(), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    merge2 = Add()([conv3, up2])
    conv6 = focusnetAlphaLayer(255)(merge2)
    
    up3 = Conv2D(144, 2, activation = LeakyReLU(), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge3 = Add()([conv2, up3])
    conv7 = focusnetAlphaLayer(144)(merge3)
    
    up4 = Conv2D(84, 2, activation = LeakyReLU(), padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge4 = Add()([conv1, up4])
    conv8 = focusnetAlphaLayer(84)(merge4)
    
    out = Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    model = Model(inputs, out)
    
    # model.summary()

    # model.compile(optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True), loss = dice_coef_loss, metrics = [dice_coef, jaccard_coef, 'acc'])

    return model


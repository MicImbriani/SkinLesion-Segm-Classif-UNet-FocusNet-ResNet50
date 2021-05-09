from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt


img_height = 256
img_width = 256
img_channels = 1


def net(x):
        
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y
    
    def attention_block(y, nb_channels_in, _strides):
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y_scores = layers.Activation('softmax')(y)
        y = add_common_layers(y)
        
        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
        y = layers.Multiply()([y_scores, y])
        y = add_common_layers(y)
        
        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        y = add_common_layers(y)
        
        return y


    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNet
        y = layers.Conv2D(nb_channels_in, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y
    
    conv1 = layers.Conv2D(84, kernel_size=(3, 3), strides=(1,1), padding='same')(x)
    conv1_d = residual_block(conv1, 84, 84, _project_shortcut=False, _strides=(2,2))
    
    conv2 = residual_block(conv1_d, 84, 144, _project_shortcut=True, _strides=(1,1))
    conv2_d = residual_block(conv2, 144, 144, _project_shortcut=False, _strides=(2,2))
    
    conv3 = residual_block(conv2_d, 144, 255, _project_shortcut=True, _strides=(1,1))
    conv3_d = residual_block(conv3, 255, 255, _project_shortcut=False, _strides=(2,2))
    
    conv4 = residual_block(conv3_d, 255, 396, _project_shortcut=True, _strides=(1,1))
    conv4_d = residual_block(conv4, 396, 396, _project_shortcut=False, _strides=(2,2))
    
    bottleneck = residual_block(conv4_d, 396, 510, _project_shortcut=True, _strides=(1,1))
    
    up1 = layers.UpSampling2D(size = (2,2))(bottleneck)
    up1_c = residual_block(up1, 510, 396, _project_shortcut=True, _strides=(1,1))
    merge1 = layers.Add()([conv4, up1_c])
    conv5 = residual_block(merge1, 396, 396, _project_shortcut=False, _strides=(1,1))
    
    up2 = layers.UpSampling2D(size = (2,2))(conv5)
    up2_c = residual_block(up2, 396, 255, _project_shortcut=True, _strides=(1,1))
    merge2 = layers.Add()([conv3, up2_c])
    conv6 = residual_block(merge2, 255, 255, _project_shortcut=False, _strides=(1,1))
    
    up3 = layers.UpSampling2D(size = (2,2))(conv6)
    up3_c = residual_block(up3, 255, 144, _project_shortcut=True, _strides=(1,1))
    merge3 = layers.Add()([conv2, up3_c])
    conv7 = residual_block(merge3, 144, 144, _project_shortcut=False, _strides=(1,1))
    
    up4 = layers.UpSampling2D(size = (2,2))(conv7)
    up4_c = residual_block(up4, 144, 84, _project_shortcut=True, _strides=(1,1))
    merge4 = layers.Add()([conv1, up4_c])
    conv8 = residual_block(merge4, 84, 48, _project_shortcut=True, _strides=(1,1))
    conv9 = residual_block(conv8, 48, 27, _project_shortcut=True, _strides=(1,1))
    conv10 = residual_block(conv9, 27, 9, _project_shortcut=True, _strides=(1,1))
    out = layers.Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv10)
    
    return out

def focusnet():
    image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    network_output = net(image_tensor)
            
    model = models.Model(inputs=[image_tensor], outputs=[network_output])

    return model
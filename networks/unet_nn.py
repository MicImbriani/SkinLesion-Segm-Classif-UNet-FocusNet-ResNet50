from keras.layers import (
    Input,
    Lambda,
    Conv2D,
    Dropout,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
)
from keras.models import Model, load_model, model_from_json

import keras.backend as K

channel_axis = 1 if K.image_data_format() == "channels_first" else -1


def conv_block(neurons, block_input, batch_norm=False, middle=False):
    conv1 = Conv2D(
        neurons,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(block_input)
    if batch_norm:
        conv1 = BatchNormalization(axis=channel_axis)(conv1)
    conv2 = Conv2D(
        neurons,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv1)
    if batch_norm:
        conv1 = BatchNormalization(axis=channel_axis)(conv1)
    if middle:
        conv2 = Dropout(0.2)(conv2)
        return conv2
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    return pool, conv2


def deconv_block(neurons, block_input, shortcut, batch_norm=False):
    deconv = Conv2D(
        neurons,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(UpSampling2D(size=(2, 2))(block_input))
    upconv = Concatenate(axis=3)([deconv, shortcut])
    upconv = Conv2D(
        neurons,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(upconv)
    if batch_norm:
        upconv = BatchNormalization(axis=channel_axis)(upconv)
    upconv = Conv2D(
        neurons,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(upconv)
    if batch_norm:
        upconv = BatchNormalization(axis=channel_axis)(upconv)
    return upconv


def unet(input_size=(256, 256, 1), batch_norm=False):
    input_layer = Input(input_size)

    # Down
    conv1, shortcut1 = conv_block(64, input_layer, batch_norm)
    conv2, shortcut2 = conv_block(128, conv1, batch_norm)
    conv3, shortcut3 = conv_block(256, conv2, batch_norm)
    conv4, shortcut4 = conv_block(512, conv3, batch_norm)

    # Middle
    convm = conv_block(1024, conv4, batch_norm, middle=True)

    # Up
    deconv4 = deconv_block(512, convm, shortcut4, batch_norm)
    deconv3 = deconv_block(256, deconv4, shortcut3, batch_norm)
    deconv2 = deconv_block(128, deconv3, shortcut2, batch_norm)
    deconv1 = deconv_block(64, deconv2, shortcut1, batch_norm)

    final_conv = Conv2D(
        2, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal"
    )(deconv1)
    output_layer = Conv2D(1, 1, activation="sigmoid")(final_conv)

    model = Model(input_layer, output_layer)

    return model

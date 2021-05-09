import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"

#from unet import unet
from networks.unet_nn import unet
from networks.focusnet_nn import focusnet
from networks.unet_res_se import unet_res_se

import metrics
#import data_process
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.metrics import MeanIoU



trainData = np.load('/var/tmp/mi714/test_new_npy2/data.npy')
trainMask = np.load('/var/tmp/mi714/test_new_npy2/dataMask.npy')

valData = np.load('/var/tmp/mi714/test_new_npy2/dataval.npy')
valMask = np.load('/var/tmp/mi714/test_new_npy2/dataMaskval.npy')

trainData = trainData.astype('float32')
mean = np.mean(trainData)  # mean for data centering
std = np.std(trainData)  # std for data normalization

valData = valData.astype('float32')

trainData -= mean
trainData /= std

valData -= mean 
valData /= std

trainMask = trainMask.astype('float32')
trainMask /= 255.  # scale masks to [0, 1]

valMask = valMask.astype('float32')
valMask /= 255.  # scale masks to [0, 1]


# CHANGE FOLDER NAMES TO "ISIC-2017_Training_Data"

###############################################################################################

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x

def random_zoom(x, y, zoom_range, row_index=1, col_index=2, channel_index=0,
                fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y


def random_rotation(x, y, rg, row_index=1, col_index=2, channel_index=0,
                    fill_mode='nearest', cval=0.):
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y

def Augmentation(X, Y):
        print('Augmentation model...')
        total = len(X)
        x_train, y_train = [], []
        
        for i in range(total):
            x, y = X[i], Y[i]
            #standart
            x_train.append(x)
            y_train.append(y)
        
#            for _ in xrange(1):
#                _x, _y = elastic_transform(x[0], y[0], 100, 20)
#                x_train.append(_x.reshape((1,) + _x.shape))
#                y_train.append(_y.reshape((1,) + _y.shape))
            
            #flip x
            x_train.append(flip_axis(x, 2))
            y_train.append(flip_axis(y, 2))
            #flip y
            x_train.append(flip_axis(x, 1))
            y_train.append(flip_axis(y, 1))
            #continue
            #zoom
            for _ in range(0):
                _x, _y = random_zoom(x, y, (0.9, 1.1))
                x_train.append(_x)
                y_train.append(_y)
            for _ in range(0):
                _x, _y = random_rotation(x, y, 5)
                x_train.append(_x)
                y_train.append(_y)
            #intentsity
#            for _ in range(0):
#                _x = random_channel_shift(x, 5.0)
#                x_train.append(_x)
#                y_train.append(y)
    
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train

trainData, trainMask = Augmentation(trainData,trainMask)

####################################################################################


model_name = "unet_nobn"

path = "/var/tmp/mi714/aug17/models/NEW_mynpy3dimensions/unet_nobn/unet_nobn2/"
os.makedirs(path, exist_ok=True)

# Selection of which model to train
model = unet(batch_norm=False)
# model = unet(batch_norm=True)
# model = unet_res_se()
# model = focusnet()

my_adam = Adam(lr=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
my_sgd = SGD(lr=0.00001, momentum=0.9, decay=1e-6, nesterov=True)

model.compile(optimizer=my_adam,
                loss=metrics.dice_coef_loss,
                metrics=[metrics.dice_coef_loss,
                        metrics.jaccard_coef_loss,
                        metrics.true_positive,
                        metrics.true_negative,
                        ])
model.summary()

callbacks = [
        ModelCheckpoint(path + "/" + model_name + "_weights.h5", 
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode='min'
                ),
        EarlyStopping(patience=20,
                monitor='val_loss',
                mode='min',
                verbose=1,
                min_delta=0.001
                ),
        ReduceLROnPlateau(monitor='val_loss',
                mode='min',
                min_delta=0.01,
                cooldown=0,
                min_lr=0.5e-7,
                factor=0.5,
                patience=5,
                verbose=1
                )]

history = model.fit(trainData,
                    trainMask,
                    batch_size=6,
                    epochs=50,
                    verbose=1,
                    validation_data=(valData, valMask),
                    shuffle=True,
                    callbacks=callbacks
                    )

model.save(path + "/" + model_name + "_model.h5')

########################################################################################################################################################################


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(path + '/loss.png')
plt.clf()
# summarize history for jaccard
plt.plot(history.history['jaccard_coef_loss'])
plt.plot(history.history['val_jaccard_coef_loss'])
plt.title('model jaccard coef loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(path + '/jaccard.png')
plt.clf()
# summarize history for positive
plt.plot(history.history['true_positive'])
plt.plot(history.history['val_true_positive'])
plt.title('model true positive')
plt.ylabel('positives')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(path + '/pos.png')
plt.clf()
# summarize history for negative
plt.plot(history.history['true_negative'])
plt.plot(history.history['val_true_negative'])
plt.title('model true negative')
plt.ylabel('negatives')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(path + '/neg.png')
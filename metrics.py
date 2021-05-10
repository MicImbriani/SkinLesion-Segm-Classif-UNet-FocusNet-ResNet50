import keras.backend as K
from keras import losses
import tensorflow as tf



def true_positive(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
    return tp 

def true_negative(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn 

def false_positive(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    return 1 - tp

def false_negative(y_true, y_pred):
    tn = true_negative(y_true, y_pred)
    return 1 - tn

def sensitivity(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp / (tp + fn)

def specificity(y_true, y_pred):
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return tn / (tn + fp)


# def dice_coef(y_true, y_pred, smooth=1):
#     intersection = K.sum(K.flatten(y_true) * K.flatten(y_pred))
#     union = K.sum(y_true) + K.sum(y_pred)
#     return K.mean( (2. * intersection + smooth) / (union + smooth))

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def dice_coef_loss(y_true, y_pred):
    dice = dice_coef(y_true, y_pred)
    return (1 - dice)

# def jaccard_coef(y_true, y_pred, smooth=1):
#     intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
#     summation = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
#     jac = (intersection + smooth) / (summation - intersection + smooth)
#     return K.mean(jac)

def jaccard_coef(y_true, y_pred):
    smooth=1
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def jaccard_coef_loss(y_true, y_pred):
    jac = jaccard_coef(y_true, y_pred)
    return (1 - jac)
    

def focal_loss(y_true, y_pred):    
    alpha = 0.8
    gamma = 2
    
    y_pred = K.flatten(y_pred)
    y_true = K.flatten(y_true)
    
    BCE = K.binary_crossentropy(y_true, y_pred)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    
    return focal_loss
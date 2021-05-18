from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.applications.resnet50 import ResNet50

def get_res():

    model = Sequential()
    #new_input = Input(shape=(256, 256, 1))
    model.add(ResNet50(include_top = False, 
                       pooling = 'avg',
                       weights = None))
                       #input_shape=(256,256,3),
                       #input_tensor=new_input,
                       #classes=2))

    # 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
    model.add(Dense(2, activation = 'softmax'))

    # Say not to train first layer (ResNet) model as it is already trained
    model.layers[0].trainable = False

    return model

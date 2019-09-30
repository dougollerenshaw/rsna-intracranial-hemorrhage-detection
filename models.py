import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras import backend as K
from keras.optimizers import SGD
from keras.optimizers import Adam


def models(model_name, input_image_size):
    model_translator = {
        'vgg': vgg_convnet(input_image_size)
    }
    return model_translator[model_name]


def vgg_convnet(input_image_size):
    # vgg like convnet from here: https://keras.io/getting-started/sequential-model-guide/
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu',
                     input_shape=(input_image_size, input_image_size, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    # replace 'softmax' with 'sigmoid' to allow probabilities not to sum to 1
    model.add(Dense(5, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )

    return model

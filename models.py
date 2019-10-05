import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras import backend as K
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.applications.inception_resnet_v2 import InceptionResNetV2


def models(model_name, input_image_size, number_of_output_categories):
    model_translator = {
        'vgg': vgg_convnet(input_image_size, number_of_output_categories),
        'inception': inception_imagenet(input_image_size, number_of_output_categories)
    }
    return model_translator[model_name]


def vgg_convnet(input_image_size, number_of_output_categories):
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
    model.add(Dense(number_of_output_categories, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['categorical_accuracy','accuracy']
    )

    return model


def inception_imagenet(input_image_size, number_of_output_categories):

    # even though this network was trained on RGB images, we can repeat our images on all three channels
    # so it will play nice

    base_model = InceptionResNetV2(include_top = False,
                                weights = "imagenet", 
                                input_shape = (input_image_size,input_image_size,3))

    base_model.trainable = False
    model = Sequential()
    model.add(base_model)

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_output_categories, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['categorical_accuracy','accuracy']
    )

    return model
import keras
import tensorflow as tf
from keras import metrics
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19


def models(model_name, input_image_size, number_of_output_categories):
    model_translator = {
        'vgg': vgg_convnet,
        'vgg19':vgg19,
        'inception_custom': inception_custom,
        'inception': inception_imagenet,
    }
    return model_translator[model_name](input_image_size, number_of_output_categories)

def vgg19(input_image_size, number_of_output_categories):
    model = VGG19(
        include_top=False, 
        weights='imagenet', 
        input_shape=(input_image_size,input_image_size,3), 
    )


    #Adding custom Layers 
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(number_of_output_categories, activation='sigmoid')(x)

    # creating the final model 
    model = Model(input = model.input, output = predictions)

    # compile the model 
    model.compile(
        loss = "binary_crossentropy", 
        optimizer = SGD(lr=0.0001, momentum=0.9), 
        metrics=["accuracy"]
    )
    
    return model


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
        metrics=['accuracy']
    )

    return model

def inception_custom(input_image_size, number_of_output_categories):
    # from https://maelfabien.github.io/deeplearning/inception/#in-keras
    # see also https://medium.com/@mannasiladittya/building-inception-resnet-v2-in-keras-from-scratch-a3546c4d93f0

    input_img = Input(shape=(input_image_size, input_image_size, 1))

    ### 1st layer
    layer_1 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
    layer_1 = Conv2D(10, (3,3), padding='same', activation='relu')(layer_1)

    layer_2 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
    layer_2 = Conv2D(10, (5,5), padding='same', activation='relu')(layer_2)

    layer_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
    layer_3 = Conv2D(10, (1,1), padding='same', activation='relu')(layer_3)

    mid1 = tf.keras.layers.concatenate([layer_1, layer_2, layer_3], axis = 3)
    flat_1 = Flatten()(mid1)

    ### 2nd layer
    layer2_1 = Conv2D(10, (1,1), padding='same', activation='relu')(flat_1)
    layer2_1 = Conv2D(10, (3,3), padding='same', activation='relu')(layer2_1)

    layer2_2 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
    layer2_2 = Conv2D(10, (5,5), padding='same', activation='relu')(layer2_2)

    layer2_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
    layer2_3 = Conv2D(10, (1,1), padding='same', activation='relu')(layer2_3)

    mid2 = tf.keras.layers.concatenate([layer2_1, layer2_2, layer2_3], axis = 3)
    flat_2 = Flatten()(mid2)

    ### dense layers
    dense_1 = Dense(1200, activation='relu')(flat_2)
    dense_2 = Dense(600, activation='relu')(dense_1)
    dense_3 = Dense(150, activation='relu')(dense_2)
    output = Dense(number_of_output_categories, activation='sigmoid')(dense_3)

    model = Model([input_img], output)

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
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
        metrics=['accuracy']
    )
    # see: https://stackoverflow.com/questions/43544358/categorical-crossentropy-need-to-use-categorical-accuracy-or-accuracy-as-the-met
    # for explanation of how keras automatically chooses accuracy type

    return model
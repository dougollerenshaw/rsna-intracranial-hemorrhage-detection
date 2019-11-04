import keras
import tensorflow as tf
from tensorflow.python.ops import array_ops
from keras import metrics
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate
from keras.layers import Activation, Dropout, Flatten, Dense, Input, BatchNormalization
from keras.optimizers import RMSprop
from keras import backend as K
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.initializers import Constant, glorot_normal
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.nasnet import NASNetLarge



def models(model_name, input_image_size, number_of_output_categories):
    model_translator = {
        'vgg': vgg_convnet,
        'vgg19': vgg19,
        'vgg_custom': vgg_custom,
        'inception_custom': inception_custom,
        'inception': inception_imagenet,
        'inception_resnetv2':inception_resnetv2,
        'nasnet_large':nasnet_large,
    }
    return model_translator[model_name](input_image_size, number_of_output_categories)


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.1, gamma=2.0):
    # https://www.kaggle.com/xhlulu/rsna-intracranial-simple-densenet-in-keras 
    # https://github.com/ailias/Focal-Loss-implement-on-Tensorflow/blob/master/focal_loss.py

    """Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


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
        optimizer = SGD(), #SGD(lr=0.0001, momentum=0.9), 
        metrics=["accuracy"]
    )
    
    return model


def vgg_custom(input_image_size, number_of_output_categories):
    # vgg like convnet from here: https://keras.io/getting-started/sequential-model-guide/
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), input_shape=(input_image_size, input_image_size, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(128, (3, 3), use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # replace 'softmax' with 'sigmoid' to allow probabilities not to sum to 1
    model.add(Dense(number_of_output_categories, 
                    kernel_initializer=glorot_normal(seed=11),
                    activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model

def inception_custom(input_image_size, number_of_output_categories):
    from keras.layers.convolutional import Conv2D, MaxPooling2D
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

    mid1 = Concatenate(axis=3)([layer_1, layer_2, layer_3])
    flat_1 = Flatten()(mid1)

    ### 2nd layer
    #layer2_1 = Conv2D(10, (1,1), padding='same', activation='relu')(mid1)
    #layer2_1 = Conv2D(10, (3,3), padding='same', activation='relu')(layer2_1)

    #layer2_2 = Conv2D(10, (1,1), padding='same', activation='relu')(input_img)
    #layer2_2 = Conv2D(10, (5,5), padding='same', activation='relu')(layer2_2)

    #layer2_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
    #layer2_3 = Conv2D(10, (1,1), padding='same', activation='relu')(layer2_3)

    #mid2 = Concatenate(axis=3)([layer2_1, layer2_2, layer2_3])
    #flat_2 = Flatten()(mid2)

    ### dense layers
    dense_1 = Dense(600, activation='relu')(flat_1)
    dense_2 = Dense(150, activation='relu')(dense_1)
    output = Dense(number_of_output_categories, activation='sigmoid')(dense_2)

    model = Model([input_img], output)

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )

    return model


def inception_imagenet(input_image_size, number_of_output_categories):
    from keras.applications.inception_resnet_v2 import InceptionResNetV2

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

def inception_resnetv2(input_image_size, number_of_output_categories):

    model = InceptionResNetV2(include_top = False,
                                weights = "imagenet", 
                                input_shape = (input_image_size,input_image_size,3))


    #Adding custom Layers 
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)
    predictions = Dense(number_of_output_categories, activation='sigmoid')(x)

    # creating the final model 
    model = Model(input = model.input, output = predictions)

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )

    return model

def nasnet_large(input_image_size, number_of_output_categories):



    base_model = NASNetLarge(include_top = False,
                                weights = "imagenet", 
                                input_shape = (input_image_size,input_image_size,3))

    base_model.trainable = False
    model = Sequential()
    model.add(base_model)

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(number_of_output_categories, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy']
    )
    
    return model

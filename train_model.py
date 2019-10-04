import pydicom

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm as progress
from skimage import exposure, io, transform
from skimage.transform import rotate, warp
from skimage.transform import SimilarityTransform
import shutil
import urllib
import warnings
import datetime

import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras import backend as K
from keras.optimizers import SGD
from keras.optimizers import Adam

from keras_preprocessing.image import ImageDataGenerator

import tensorflow as tf

import utils
from models import models

def make_y_image(generator,model,filename):
    generator.__reset__()
    X,y=generator.__next__()

    y_pred = model.predict_proba(X)
    fig,ax=plt.subplots(1,2,figsize=(10,10))
    ax[0].imshow(y,aspect='auto')
    ax[1].imshow(y_pred,aspect='auto')
    ax[0].set_title('y_actual')
    ax[1].set_title('y_pred')
    fig.savefig(filename)

def main(dataloc = r'D:\rsna-intracranial-hemorrhage-detection'):
    # load training df
    tdf = utils.load_training_data(dataloc)

    # set up training fraction
    ## train and validate dataframes
    training_fraction = 0.15
    shuff = tdf.sample(frac=training_fraction)
    train_df = shuff.iloc[:int(0.90*len(shuff))]
    validate_df = shuff.iloc[int(0.10*len(shuff)):]
    len(shuff),len(train_df),len(validate_df)

    batch_size = 16
    desired_size = 256

    # set up generators
    categories = utils.define_categories(include_any=True)
    
    train_generator = utils.Dicom_Image_Generator(
        train_df.reset_index(),
        ycols=categories,
        desired_size=desired_size,
        batch_size=batch_size,
        random_transform=False
    )

    validate_generator = utils.Dicom_Image_Generator(
        validate_df.reset_index(),
        ycols=categories,
        desired_size=desired_size,
        batch_size=batch_size,
        random_transform=False
    )


    # load model
    model = models('vgg', input_image_size=desired_size, number_of_output_categories=len(categories))

    # #load weights (optional)
    #model.load_weights("model_weights_6_outputs_iteration_CRASH_DUMP=0_2019-10-01 19:52:34.598103.h5")

    # train

    for i in range(10):
        try:
            model.fit_generator(
                generator=train_generator,
                steps_per_epoch=len(train_df)//batch_size,
                validation_data=validate_generator,
                validation_steps=len(validate_df)//batch_size,
                epochs=15
            )
            model.save_weights("model_weights_6_outputs_iteration={}_{}.h5".format(i,str(datetime.datetime.now())))
            y_image_filename = os.path.join(dataloc,'y_plot_validate_{}.png'.format(str(datetime.datetime.now())))
            make_y_image(validate_generator,model,y_image_filename)
        except Exception as e:
            print(e)
            model.save_weights("model_weights_6_outputs_iteration_CRASH_DUMP={}_{}.h5".format(i,str(datetime.datetime.now())))

if __name__ == '__main__':
    main()

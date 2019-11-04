import pydicom

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import exposure, io, transform
from skimage.transform import rotate, warp
from skimage.transform import SimilarityTransform
import urllib
import warnings
import datetime
import fnmatch

import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop
from keras import backend as K
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

from keras_preprocessing.image import ImageDataGenerator

import tensorflow as tf

import utils
from models import models


class rsna_model(object):

    def __init__(self,
                dataloc = r'D:\rsna-intracranial-hemorrhage-detection', 
                weights_path = None,
                model_name = 'vgg',
                training_fraction = 0.15,
                batch_size = 16,
                img_size = 256,
                epochs = 1,
                rgb = False,
                old_equalize = True,
                random_transform=False,
                random_state=0,
                ):

        self.dataloc = dataloc
        self.weights_path = weights_path
        self.model_name = model_name
        self.training_fraction = training_fraction
        self.batch_size = batch_size
        self.img_size = img_size
        self.epochs = epochs
        self.rgb = rgb
        self.old_equalize = old_equalize
        self.random_transform = random_transform
        self.random_state = random_state

        self.datestamp = str(datetime.datetime.now()).replace(':','_').replace(' ','T')

    def make_y_image(self,generator,model,filename):
        generator.__reset__()
        X,y=generator.__next__()

        y_pred = model.predict_proba(X)
        fig,ax=plt.subplots(1,2,figsize=(10,10))
        ax[0].imshow(y,aspect='auto')
        ax[1].imshow(y_pred,aspect='auto')
        ax[0].set_title('y_actual')
        ax[1].set_title('y_pred')
        fig.savefig(filename)

    def build(self):

        # load training df
        self.tdf = utils.load_training_data(self.dataloc)

        # drop missing image
        drop_idx = [i for i,row in self.tdf['filename'].iteritems() if fnmatch.fnmatch(row,'*ID_33fcf4d99*')]
        self.tdf = self.tdf.drop(drop_idx)
        
        # set up training fraction
        ## train and validate dataframes
        shuff = self.tdf.sample(frac=self.training_fraction, random_state=self.random_state)
        self.train_df = shuff.iloc[:int(0.90*len(shuff))]
        self.validate_df = shuff.iloc[int(0.90*len(shuff)):]
        len(shuff),len(self.train_df),len(self.validate_df)

        # set up generators
        self.categories = utils.define_categories(self.train_df)
        
        self.train_generator = utils.Three_Channel_Generator(
                                        self.train_df.reset_index(),
                                        ycols=self.categories,
                                        desired_size=self.img_size,
                                        batch_size=self.batch_size,
                                        random_transform=self.random_transform,
                                        rgb=True)

        self.validate_generator = utils.Three_Channel_Generator(
                                        self.validate_df.reset_index(),
                                        ycols=self.categories,
                                        desired_size=self.img_size,
                                        batch_size=self.batch_size,
                                        random_transform=False,
                                        rgb=True)

        # load model
        self.model = models(self.model_name, 
                                input_image_size=self.img_size, 
                                number_of_output_categories=len(self.categories))

        if self.weights_path is not None:
            self.model.load_weights(self.weights_path)

        # setup callbacks
        earlystop = EarlyStopping(patience=10)

        learning_rate_reduction = ReduceLROnPlateau(
                                                    monitor='categorical_accuracy', 
                                                    patience=2, 
                                                    verbose=1, 
                                                    factor=0.5, 
                                                    min_lr=0.00001)

        checkpoint_name = os.path.join(self.dataloc,"model_weights_vgg19_{}.h5".format(self.datestamp))
        checkpoint = ModelCheckpoint(
                                    checkpoint_name, 
                                    monitor='val_acc', 
                                    verbose=0, 
                                    save_best_only=True, 
                                    save_weights_only=True,
                                    mode='auto')

        self.callbacks = [earlystop, checkpoint]

    def train(self):
        
        self.history = self.model.fit_generator(
                        generator=self.train_generator,
                        steps_per_epoch=len(self.train_df)//self.batch_size,
                        validation_data=self.validate_generator,
                        validation_steps=len(self.validate_df)//self.batch_size,
                        epochs=self.epochs,
                        callbacks = self.callbacks,
                        verbose = 1)

    def save(self):
            
        self.model.save_weights(os.path.join(dataloc,"model_weights_{}.h5".format(self.datestamp)))
        y_image_filename = os.path.join(dataloc,'y_plot_validate_{}.png'.format(self.datestamp))
        self.make_y_image(self.validate_generator,self.model,y_image_filename)


if __name__ == '__main__':
    dataloc = '/ssd1'
    model = rsna_model(
        dataloc = '/ssd1', 
        weights_path = os.path.join(dataloc,'model_weights_vgg19_2019-11-02T16_15_53.667387.h5'),
        model_name = 'vgg19',
        training_fraction = 1,
        batch_size = 8,
        img_size = 512,
        epochs = 60,
        rgb = True,
        old_equalize = False,
        random_transform=True,
        random_state=1,
    )
    model.build()
    model.train()

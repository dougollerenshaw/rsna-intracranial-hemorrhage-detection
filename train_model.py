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
                df = None,
                weights_path = None,
                model_name = 'vgg',
                training_fraction = 0.15,
                batch_size = 16,
                img_size = 256,
                epochs = 1,
                rgb = False,
                old_equalize = True,
                class_weights = None,
                random_transform = False
                ):

        self.dataloc = dataloc
        self.df = df
        self.weights_path = weights_path
        self.model_name = model_name
        self.training_fraction = training_fraction
        self.batch_size = batch_size
        self.img_size = img_size
        self.epochs = epochs
        self.rgb = rgb
        self.old_equalize = old_equalize
        self.class_weights = class_weights
        self.random_transform = random_transform

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

        if isinstance(self.df, pd.DataFrame):
            self.tdf = self.df
        else:
            # load training df from .csv
            self.tdf = utils.load_training_data(self.dataloc)

        # drop missing image
        drop_idx = [i for i,row in self.tdf['filename'].iteritems() if fnmatch.fnmatch(row,'*ID_33fcf4d99*')]
        self.tdf = self.tdf.drop(drop_idx)
        drop_idx2 = [i for i,row in self.tdf['filename'].iteritems() if fnmatch.fnmatch(row,'*ID_6431af929*')]
        self.tdf = self.tdf.drop(drop_idx2)
        
        # set up training fraction
        ## train and validate dataframes
        shuff = self.tdf.sample(frac=self.training_fraction)
        self.train_df = shuff.iloc[:int(0.90*len(shuff))]
        self.validate_df = shuff.iloc[int(0.90*len(shuff)):]
        len(shuff),len(self.train_df),len(self.validate_df)

        # set up generators
        self.categories = utils.define_categories(self.tdf, include_any=False)
        
        self.train_generator = utils.Dicom_Image_Generator(
                                        self.train_df.reset_index(),
                                        ycols=self.categories,
                                        desired_size=self.img_size,
                                        batch_size=self.batch_size,
                                        random_transform=self.random_transform,
                                        rgb=self.rgb,
                                        old_equalize = self.old_equalize)

        self.validate_generator = utils.Dicom_Image_Generator(
                                        self.validate_df.reset_index(),
                                        ycols=self.categories,
                                        desired_size=self.img_size,
                                        batch_size=self.batch_size,
                                        random_transform=self.random_transform,
                                        rgb=self.rgb,
                                        old_equalize = self.old_equalize)

        # load model
        self.model = models(self.model_name, 
                                input_image_size=self.img_size, 
                                number_of_output_categories=len(self.categories))

        if self.weights_path is not None:
            self.model.load_weights(self.weights_path)

        # setup callbacks
        earlystop = EarlyStopping(patience=10)

        learning_rate_reduction = ReduceLROnPlateau(
                                                    monitor='val_loss', 
                                                    patience=1, 
                                                    verbose=1, 
                                                    factor=0.5, 
                                                    min_lr=0.00001,
                                                    mode='min')

        checkpoint_name = "model_weights_outputs_iteration_{}_{}.h5".format(str(self.categories), self.datestamp)
        checkpoint_path = os.path.join(self.dataloc, checkpoint_name)
        checkpoint = ModelCheckpoint(
                                    checkpoint_path, 
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    save_weights_only=False,
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
                        class_weight = self.class_weights,
                        verbose = 1)

    def save(self):
            
        self.model.save_weights("../untracked_files/model_weights_6_outputs_iteration={}.h5".format(self.datestamp))
        y_image_filename = '../untracked_files/y_plot_validate_{}.png'.format(self.datestamp)
        self.make_y_image(self.validate_generator,self.model,y_image_filename)


if __name__ == '__main__':
    model = rsna_model(dataloc = '/mnt/win_f/rsna_data',
                        weights_path = "../untracked_files/model_weights_6_outputs_iteration=0_2019-10-04 05:38:23.464537.H5")
    model.build()
    model.train()

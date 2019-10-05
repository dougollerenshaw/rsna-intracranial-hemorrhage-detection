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

from keras_preprocessing.image import ImageDataGenerator

import tensorflow as tf


class Dicom_Image_Generator:
    '''adapted from https://www.kaggle.com/kwisatzhaderach/dicom-generator'''

    def __init__(self, df, ycols, subset='train', batch_size=12, desired_size=512, random_transform=True, rgb=False):
        self.df = df
        self.length = len(df)
        self.subset = subset
        self.batch_size = batch_size
        self.position = 0
        self.desired_size = desired_size
        self.ycols = ycols
        self.random_transform = random_transform
        self.rgb = rgb

    def __iter__(self):
        return self

    def open_image(self, filename):
        ds = pydicom.dcmread(filename)
        im = ds.pixel_array
        return im/im.max()

    def apply_random_transform(self, input_image):
        rotation = np.random.uniform(low=-10, high=10)
        horizontal_translation = np.random.uniform(low=-10, high=10)
        vertical_translation = np.random.uniform(low=-10, high=10)
        tform = SimilarityTransform(translation=(
            horizontal_translation, vertical_translation))

        rotated_image = rotate(input_image, rotation,
                               cval=np.mean(input_image[:10, :10]))
        translated_image = warp(rotated_image, tform,
                                cval=np.mean(input_image[:10, :10]))
        return translated_image

    def __reset__(self):
        self.position = 0

    def __next__(self):
        if self.rgb:
            X, y = np.empty((self.batch_size, self.desired_size,
                         self.desired_size, 3)), []
        else:
            X, y = np.empty((self.batch_size, self.desired_size,
                            self.desired_size, 1)), []

        for i in range(self.batch_size):
            filepath = self.df.iloc[self.position]['filename']

            # there is a missing file, skip this
            if fnmatch.fnmatch(filepath, '*ID_33fcf4d99*'):
                continue

            image = self.open_image(filepath)

            # occasionally image sizes in this dataset vary
            if (image.shape[0] != self.desired_size):
                image = transform.resize(
                    image, (self.desired_size, self.desired_size))

            if self.random_transform:
                image = self.apply_random_transform(image)
                lr = np.random.choice([-1, 1])
                image = image[:, ::lr]  # flip image l/r

            try:
                image = exposure.equalize_hist(image)
            except:
                image = np.zeros_like(image)
                warnings.warn(
                    'image {} could not be equalized, returning as array of zeros'.format(filepath))

            if self.rgb:
                X[i] = np.expand_dims(np.repeat(
                    image[..., np.newaxis], 3, -1), axis=0).astype(float)
            else:
                X[i] = np.expand_dims(np.expand_dims(
                    image, axis=0), axis=3).astype(float)

            y.append(self.df.iloc[self.position][self.ycols])
            self.position += 1
            if (self.position >= self.length):
                self.position = 0
        if (self.subset == 'test'):
            return X
        else:
            return (X, np.asarray(y).astype(int))


def define_categories(include_any=False):
    categories = [
        'epidural',
        'intraparenchymal',
        'intraventricular',
        'subarachnoid',
        'subdural',
    ]
    if include_any:
        categories = categories + ['any']
    return categories

def make_y_image(generator, model, filename):
    generator.__reset__()
    X, y = generator.__next__()

    y_pred = model.predict_proba(X)
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(y, aspect='auto')
    ax[1].imshow(y_pred, aspect='auto')
    ax[0].set_title('y_actual')
    ax[1].set_title('y_pred')
    fig.savefig(filename)

def load_training_data(dataloc):

    traindf = pd.read_csv(os.path.join(dataloc,'stage_1_train.csv'))

    traindf['type'] = traindf['ID'].map(lambda x:x.split('_')[2])

    traindf['filename'] = traindf['ID'].map(lambda x:os.path.join(dataloc,'stage_1_train_images',('ID_' + x.split('_')[1] + '.dcm')))

    traindf['ID'] = traindf['ID'].map(lambda x:'ID_'+x.split('_')[1])

    ## pivot

    tdf_all = traindf[['Label', 'ID', 'type','filename']].drop_duplicates().pivot(
        index='filename', columns='type', values='Label').reset_index()
    tdf_all.index.rename('index',inplace=True)
    positive_examples = tdf_all.query('any == 1')
    # positive_examples.drop(columns='any',inplace=True)

    categories = [c for c in positive_examples.columns if c != 'filename']

    null_examples = tdf_all.query('any == 0')
    # null_examples.drop(columns='any',inplace=True)

    tdf = pd.concat([positive_examples,null_examples.sample(len(positive_examples),random_state=0)])

    return tdf

def load_test_data(dataloc):
    test_filenames = os.listdir(os.path.join(dataloc,'stage_1_test_images'))
    d = {'filename':test_filenames}
    categories = define_categories(include_any=True)
    d.update({cat:np.zeros(len(test_filenames)) for cat in categories})
    testdf = pd.DataFrame(d)
    testdf['ID'] = testdf['filename'].map(lambda x:'ID_'+x.split("_")[1][:-4])
    testdf['filename'] = testdf['filename'].map(lambda x:os.path.join(dataloc,'stage_1_test_images',('ID_' + x.split('_')[1])))

    return testdf

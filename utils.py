import pydicom

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
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


class Dicom_Image_Generator():
    '''adapted from https://www.kaggle.com/kwisatzhaderach/dicom-generator'''

    def __init__(self, 
                    df, 
                    ycols, 
                    rgb,
                    old_equalize = True,
                    subset='train', 
                    batch_size=12, 
                    desired_size=512, 
                    random_transform=True, 
                    ):
        self.df = df
        self.length = len(df)
        self.subset = subset
        self.batch_size = batch_size
        self.position = 0
        self.desired_size = desired_size
        self.ycols = ycols
        self.random_transform = random_transform
        self.rgb = rgb
        self.old_equalize = old_equalize

    def __iter__(self):
        return self

    def window_image(self, img, data):

        dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
                    
        center,width,intercept,slope = [self.get_field_as_int(x) for x in dicom_fields]

        img = (img*slope + intercept)
        img_max = center + width//2
        img_min = center - width//2
        img[img>img_max] = img_min
        img = img - img_min
        img = img / img_max 
        # zap any nans that somehow showup
        img[np.where(np.isnan(img), True, False)] = img_min
        return img

    def get_field_as_int(self, x):
        #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
        if type(x) == pydicom.multival.MultiValue:
            return int(x[0])
        else:
            return int(x)

    def load_scale_image(self, filename):

        ds = pydicom.dcmread(filename)
        im = ds.pixel_array

        if self.old_equalize:
            # https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
            # http://www.janeriksolem.net/histogram-equalization-with-python-and.html
            im = im / im.max()            
            im = exposure.equalize_hist(im)
        else:
            im = self.window_image(im, ds)
            im = exposure.equalize_hist(im)
        return im

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

            image = self.load_scale_image(filepath)

            # occasionally image sizes in this dataset vary
            if (image.shape[0] != self.desired_size):
                image = transform.resize(
                    image, (self.desired_size, self.desired_size))

            if self.random_transform:
                image = self.apply_random_transform(image)
                lr = np.random.choice([-1, 1])
                image = image[:, ::lr]  # flip image l/r

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

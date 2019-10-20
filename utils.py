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


class Three_Channel_Generator():
    '''adapted from https://www.kaggle.com/kwisatzhaderach/dicom-generator'''

    def __init__(self, 
                    df, 
                    ycols, 
                    subset='train', 
                    batch_size=12, 
                    desired_size=512, 
                    random_transform=False,
                    rgb=True
                    ):
        self.df = df
        self.length = len(df)
        self.subset = subset
        self.batch_size = batch_size
        self.position = 0
        self.desired_size = desired_size
        self.ycols = ycols
        self.random_transform = random_transform

    def __iter__(self):
        return self
    
    def rescale_image(self, image, slope, intercept):
        return image * slope + intercept

    def apply_window(self, image, center, width):
        image = image.copy()
        min_value = center - width // 2
        max_value = center + width // 2
        image[image < min_value] = min_value
        image[image > max_value] = max_value
        return image
    
    def apply_random_transform(self, input_image, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        rotation = np.random.uniform(low=-10, high=10)
        horizontal_translation = np.random.uniform(low=-10, high=10)
        vertical_translation = np.random.uniform(low=-10, high=10)
        tform = SimilarityTransform(translation=(
            horizontal_translation, vertical_translation))

        rotated_image = rotate(input_image, rotation,
                               cval=np.mean(input_image[:10, :10]))
        translated_image = warp(rotated_image, tform,
                                cval=np.mean(input_image[:10, :10]))
        lr = np.random.choice([-1, 1])
        return translated_image[:, ::lr] # flip image l/r
#         return rotated_image # flip image l/r

    def open_scale_image(self, data):

        im = data.pixel_array
        
        shape = im.shape[0]
        image = np.empty((shape,shape,3))
        
        dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
                    
        center,width,intercept,slope = [self.get_field_as_int(x) for x in dicom_fields]

        rescaled_image = self.rescale_image(im,slope,intercept)
        image1 = self.apply_window(rescaled_image, 40, 80) # brain
        image2 = self.apply_window(rescaled_image, 80, 200) # subdural
        image3 = exposure.equalize_hist(rescaled_image)
        
        
        image1 = (image1 - 0) / 80
        image2 = (image2 - (-20)) / 200
        image3 = (image3 - image3.min()) / (image3.max()-image3.min())
        
        if self.random_transform:
            seed = np.random.randint(2**16)
            image1 = self.apply_random_transform(image1,random_state=seed)
            image2 = self.apply_random_transform(image2,random_state=seed)
            image3 = self.apply_random_transform(image3,random_state=seed)
        
        image = np.array([
            image1, # - image1.mean(),
            image2, # - image2.mean(),
            image3, # - image3.mean(),
        ]).transpose(1,2,0)

        return image

    def get_field_as_int(self, x):
        #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
        if type(x) == pydicom.multival.MultiValue:
            return int(x[0])
        else:
            return int(x)

    def load_image(self, filename):

        try:
            ds = pydicom.dcmread(filename)
            return self.open_scale_image(ds)
        except:
            warnings.warn('failed to load {}, returning zeros'.format(filename))
            return np.zeros(512,512,3)

    def __reset__(self):
        self.position = 0

    def __next__(self):
        X, y = np.empty((self.batch_size, self.desired_size,
                     self.desired_size, 3)), []

        for i in range(self.batch_size):
            filepath = self.df.iloc[self.position]['filename']

            image = self.load_image(filepath)

            # occasionally image sizes in this dataset vary
            if (image.shape[0] != self.desired_size):
                image = transform.resize(
                    image, (self.desired_size, self.desired_size))

            X[i,:,:,:] = image


            y.append(self.df.iloc[self.position][self.ycols])
            self.position += 1
            if (self.position >= self.length):
                self.position = 0
        if (self.subset == 'test'):
            return X
        else:
            return (X, np.asarray(y).astype(int))


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

        # get metadata from the first image so we don't have to do this again
        self.get_window_params(pydicom.dcmread(self.df.iloc[0]['filename']))

    def __iter__(self):
        return self
    
    def apply_window(image, center, width):
        image = image.copy()
        min_value = center - width // 2
        max_value = center + width // 2
        image[image < min_value] = min_value
        image[image > max_value] = max_value
        return image

    def get_window_params(self, data):

        dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
                    
        self.center,self.width,self.intercept,self.slope = [self.get_field_as_int(x) for x in dicom_fields]

    def window_image(self, img):

        img = (img*self.slope + self.intercept)
        img_max = (self.center + self.width//2) * 2 # scale max so get bone information needed for subdural hems
        img_min = self.center - self.width//2
        img[img>img_max] = img_max
        img[img<img_min] = img_min
        img = img - img_min
        img = img / img_max 
        # zap any nans that somehow showup
        img[np.isnan(img)] = img_min
        return img

    def get_field_as_int(self, x):
        #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
        if type(x) == pydicom.multival.MultiValue:
            return int(x[0])
        else:
            return int(x)

    def load_image(self, filename):

        try:
            ds = pydicom.dcmread(filename)
            im = ds.pixel_array
        except:
            warnings.warn('failed to load {}, returning zeros'.format(filename))
            return np.zeros(512,512)

        if self.old_equalize:
            # https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_hist
            # http://www.janeriksolem.net/histogram-equalization-with-python-and.html

            im = im / im.max()            
            try:
                im = exposure.equalize_hist(im)
            except:
                im = np.zeros_like(im)
                warnings.warn(
                    'image {} could not be equalized, returning as array of zeros'.format(filename))
        else:
            im = self.window_image(im)
            #im = exposure.equalize_hist(im)
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

            data = self.load_image(filepath)

            try:
                image = self.scale_image(data)
            except:
                print('error image ' + str(filepath))

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
                    image[..., np.newaxis], 3, -1), axis=0).astype('float32')
            else:
                X[i] = np.expand_dims(np.expand_dims(
                    image, axis=0), axis=3).astype('float32')

            y.append(self.df.iloc[self.position][self.ycols])
            self.position += 1
            if (self.position >= self.length):
                self.position = 0
        if (self.subset == 'test'):
            return X
        else:
            return (X, np.asarray(y).astype(int))


def define_categories(df, include_any=False):

    categories = [c for c in df.columns if c != 'filename']
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


def split_df_by_categories(dataloc):

    traindf = pd.read_csv(os.path.join(dataloc,'stage_1_train.csv'))

    traindf['type'] = traindf['ID'].map(lambda x:x.split('_')[2])

    traindf['filename'] = traindf['ID'].map(lambda x:os.path.join(dataloc,'stage_1_train_images',('ID_' + x.split('_')[1] + '.dcm')))

    traindf['ID'] = traindf['ID'].map(lambda x:'ID_'+x.split('_')[1])

    ## pivot

    tdf_all = traindf[['Label', 'ID', 'type','filename']].drop_duplicates().pivot(
        index='filename', columns='type', values='Label').reset_index()
    tdf_all.index.rename('index',inplace=True)

    categories = [c for c in tdf_all.columns if c != 'filename']

    tdf_by_cat = {}
    for c, category in enumerate(categories):
        cat_pos = tdf_all[tdf_all[category]==1][['filename', category]]
        cat_neg = tdf_all[tdf_all[category]==0][['filename', category]]
        # dont sample to balance here - this should be done by the generator at each epoch!
        tdf_by_cat[category] = pd.concat([cat_pos,cat_neg])

    return tdf_by_cat


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

    categories = [c for c in tdf_all.columns if c != 'filename']

    null_examples = tdf_all.query('any == 0')

    tdf = pd.concat([positive_examples,null_examples.sample(len(positive_examples),random_state=0)])

    return tdf

def load_test_data(dataloc):
    test_filenames = os.listdir(os.path.join(dataloc,'stage_1_test_images'))
    d = {'filename':test_filenames}
    categories = define_categories(include_any=False)
    d.update({cat:np.zeros(len(test_filenames)) for cat in categories})
    testdf = pd.DataFrame(d)
    testdf['ID'] = testdf['filename'].map(lambda x:'ID_'+x.split("_")[1][:-4])
    testdf['filename'] = testdf['filename'].map(lambda x:os.path.join(dataloc,'stage_1_test_images',('ID_' + x.split('_')[1])))

    return testdf

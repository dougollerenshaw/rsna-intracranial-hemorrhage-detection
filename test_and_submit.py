#import click
import utils
from models import models
import warnings
import datetime
import os
import numpy as np
import pandas as pd


def create_test_generator(testdf, categories, batch_size):
    categories = utils.define_categories(testdf, include_any=False)

    test_generator = utils.Three_Channel_Generator(
        testdf[['filename']+categories],
        ycols=categories,
        desired_size=512,
        batch_size=batch_size,
        subset='test',
        random_transform=False,
        rgb=True
    )
    return test_generator


def build_submission(testdf, y_pred, dataloc):

    # build output dataframe
    df_output = testdf.copy()

    categories = utils.define_categories(df_output)

    if len(y_pred) < len(df_output):
        mismatch = len(df_output) - len(y_pred)
        # this is necessary because the number of test images isn't evenly divisible by the batch size.
        # The predict generator stops early. I should return to this and fix it!!!
        warnings.warn(
            'y_pred is {} entries too short. Filling with zeros'.format(mismatch))
        for _ in range(mismatch):
            y_pred = np.vstack((y_pred, np.zeros_like(categories)))

    # populate columns of df_output with predictions
    for ii, cat in enumerate(categories):
        df_output[cat] = y_pred[:len(df_output), ii]


    # using the sample submission as the prototype, iterate through and fill with actual predictions
    df_output.set_index('ID', inplace=True)
    sample_submission = pd.read_csv(os.path.join(
        dataloc, 'stage_1_sample_submission.csv'))
    submission = sample_submission.copy()

    for idx, row in submission.iterrows():
        img_id = 'ID_'+row['ID'].split('_')[1]
        if img_id == 'ID_ffffcbff8':
            continue
        hem_type = row['ID'].split('_')[2]
        submission.at[idx, 'Label'] = df_output.at[img_id, hem_type]

    datestamp = str(datetime.datetime.now()).replace(':','_').replace(' ','T')
    submission_filename = os.path.join(dataloc,'submission_{}.csv'.format(datestamp))
    submission.to_csv(submission_filename, index=False)
    return submission_filename


def upload_submission(path_to_submission, message='none'):
    from urllib.request import urlretrieve
    import yaml
    credentials_url = 'https://www.dropbox.com/s/tjs6z6pna1get6g/kaggle_credentials.yml?dl=1'
    urlretrieve(credentials_url, 'credentials.yml')
    with open('credentials.yml', 'r') as f:
        credentials = yaml.load(f)
    # remove credentials so they don't end up being accidentally committed
    os.remove('credentials.yml')

    os.environ["KAGGLE_USERNAME"] = credentials["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = credentials["KAGGLE_KEY"]

    os.system('export -p | grep KAGGLE_')

    os.system('kaggle competitions submit -c rsna-intracranial-hemorrhage-detection -f "{}" -m "{}"'.format(
        path_to_submission,
        message
    ))

    os.system(
        'kaggle competitions submissions -c rsna-intracranial-hemorrhage-detection')


# @click.command()
# @click.option('--dataloc', prompt='path to data', help='location of data')
# @click.option('--weights_path', prompt='path to pretrained weights', help='location of weights h5 file')
# @click.option('--model', default='vgg', help='Name of model (must match weights)')
# @click.option('--batch_size', default=16, help='batch size for training/testing')
def main(dataloc, path_to_weights, model='vgg', batch_size=8, rgb=False):
    
    # load test data
    test_df = utils.load_test_data(dataloc)

#     categories = utils.define_categories(test_df)
    categories = [
        'any',
        'epidural',
        'intraparenchymal',
        'intraventricular',
        'subarachnoid',
        'subdural'
    ]
    print("CATEGORIES = {}".format(categories))

    # load model
    model = models(model, input_image_size=512, number_of_output_categories=len(categories))

    # load weights
    model.load_weights(path_to_weights)

    # instantiate generator
    
    test_generator = create_test_generator(test_df, categories, batch_size)

    test_generator.__reset__() # make sure the generator is starting at index 0!!!
    # predict
    y_pred = model.predict_generator(
        test_generator,
        steps=len(test_df)//batch_size+1,
        verbose=1
    )

    # build submission
    submission_filename = build_submission(test_df, y_pred, dataloc)

    print('uploading submission...')
    
    # submit response
    upload_submission(submission_filename)

if __name__ == '__main__':
    dataloc = '/ssd1'
    main(
        dataloc = dataloc,
        path_to_weights = os.path.join(dataloc,'model_weights_vgg19_2019-11-02T20_35_40.895897.h5'),
        model = 'vgg19',
        rgb=True
    )

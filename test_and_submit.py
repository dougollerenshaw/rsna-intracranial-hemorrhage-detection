import click
import utils
from models import models
import warnings
import datetime
import os


def create_test_generator(testdf, categories):
    categories = utils.define_categories()

    test_generator = utils.Dicom_Image_Generator(
        testdf[['filename']+categories],
        ycols=categories,
        desired_size=512,
        batch_size=batch_size,
        subset='test'
    )
    return test_generator


def build_submission(testdf, y_pred, dataloc):

    # build output dataframe
    df_output = testdf.copy()

    if len(y_pred) < len(df_output):
        mismatch = len(df_output) - len(y_pred)
        # this is necessary because the number of test images isn't evenly divisible by the batch size.
        # The predict generator stops early. I should return to this and fix it!!!
        warnings.warn(
            'y_pred is {} entries too short. Filling with zeros'.format(mismatch))
        for _ in range(mismatch):
            y_pred = np.vstack((y_pred, [0, 0, 0, 0, 0]))

        # populate columns of df_output with predictions
        for ii, cat in enumerate(categories):
            df_output[cat] = y_pred[:, ii]

    def get_all_prob(row):
        return np.min((1, row[categories].max()))
    df_output['any'] = df_output[categories].apply(get_all_prob, axis=1)

    # using the sample submission as the prototype, iterate through and fill with actual predictions
    df_output.set_index('ID', inplace=True)
    sample_submission = pd.read_csv(os.path.join(
        dataloc, 'stage_1_sample_submission.csv'))
    submission = sample_submission.copy()

    for idx, row in progress(submission.iterrows()):
        img_id = 'ID_'+row['ID'].split('_')[1]
        hem_type = row['ID'].split('_')[2]
        submission.at[idx, 'Label'] = df_output.at[img_id, hem_type]

    submission_filename = os.path.join(dataloc, str(datetime.datetime.now()))
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


@click.command()
@click.option('--dataloc', prompt='path to data', help='location of data')
@click.option('--weights_path', prompt='path to pretrained weights', help='location of weights h5 file')
@click.option('--model', default='vgg', help='Name of model (must match weights)')
@click.option('--batch_size', default=16, help='batch size for training/testing')
def main(dataloc, path_to_weights, model, batch_size):
    # load test data
    test_df = utils.load_test_data(dataloc)

    # load model
    model = models('vgg', input_image_size=512)

    # load weights
    model.load_weights(path_to_weights)

    # instantiate generator
    test_generator = create_test_generator(testdf, categories)

    # predict
    y_pred = model.predict_generator(
        test_generator,
        steps=len(testdf)//batch_size,
        verbose=1
    )

    # build submission
    submission_filename = build_submission(testdf, y_pred, dataloc)

    # submit response
    upload_submission(submission_filename)

if __name__ == '__main__':
    main()

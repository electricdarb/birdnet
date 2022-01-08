import boto3
from CONFIG import KEY_ID, SECRET_KEY
import os
import datetime
import numpy as np

def make_runname(prefix = None):
    if isinstance(prefix, str):
        return f'{prefix}_{datetime.datetime.now(datetime.timezone.utc).strftime("%y%m%d%H%M%S")}'
    else: 
        return datetime.datetime.now(datetime.timezone.utc).strftime("%y%m%d%H%M%S")


def save_model(runname, model, bucketname = 'bradfordgillbirddatabucket', foldername = 'modellogs'):
    filename = f'{runname}.h5'
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    path = os.path.join(foldername, filename)

    session = boto3.Session(
            aws_access_key_id=KEY_ID,
            aws_secret_access_key=SECRET_KEY)
    s3 = session.resource('s3')
    model.save(path)

    result = s3.Bucket(bucketname).upload_file(path, path)

def load_model(runname, bucketname = 'bradfordgillbirddatabucket', foldername = 'modellogs'):
    from tensorflow.keras.models import load_model

    filename = f'{runname}.h5'
    path = os.path.join(foldername, filename)

    if not os.path.exists(path):
        session = boto3.Session(
                aws_access_key_id=KEY_ID,
                aws_secret_access_key=SECRET_KEY)
        s3 = session.resource('s3')
        results = s3.Bucket(bucketname).download_file(path, path)

    return load_model(path)

def log_history(runname, history, bucketname = 'bradfordgillbirddatabucket', foldername = 'historylogs'):
    filename = f'{runname}.npy'
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    path = os.path.join(foldername, filename)

    np.save(path, history.history)

    session = boto3.Session(
            aws_access_key_id=KEY_ID,
            aws_secret_access_key=SECRET_KEY)

    s3 = session.resource('s3')
    result = s3.Bucket(bucketname).upload_file(path, path)

def get_history(runname,  bucketname = 'bradfordgillbirddatabucket', foldername = 'historylogs'):
    filename = f'{runname}.npy'
    path = os.path.join(foldername, filename)

    if not os.path.exists(path):
        if not os.path.exists(foldername):
            os.mkdir(foldername)

        session = boto3.Session(
                aws_access_key_id=KEY_ID,
                aws_secret_access_key=SECRET_KEY)

        s3 = session.resource('s3')
        result = s3.Bucket(bucketname).download_file(path, path)

    return np.load(path, allow_pickle = True)

if __name__ == '__main__':
    pass
   
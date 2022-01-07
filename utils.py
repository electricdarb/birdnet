import boto3
from CONFIG import KEY_ID, SECRET_KEY
import numpy as np
import os
import datetime

def make_runname(prefix):
    return f'{prefix}_{datetime.datetime.now(datetime.timezone.utc).strftime("%y%m%d%H%M%S")}'

def save_model(runname, model, bucketname = 'bradfordgillbirddatabucket', foldername = 'modellogs'):
    filename = f'{runname}.h5'
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

if __name__ == '__main__':
    print(make_runname('test'))
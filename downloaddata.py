import boto3
import os
from CONFIG import KEY_ID, SECRET_KEY

def downloadDirectoryFroms3(bucketName, remoteDirectoryName):
    session = boto3.Session(
            aws_access_key_id=KEY_ID,
            aws_secret_access_key=SECRET_KEY)
    s3 = session.resource('s3')
    bucket = s3.Bucket(bucketName) 
    print('Starting!')
    i = 0
    for obj in bucket.objects.filter(Prefix = remoteDirectoryName):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key)
        if i % 100 == 0:
            print(i/11500)
        i += 1
    

if __name__ == "__main__":
    downloadDirectoryFroms3('bradfordgillbirddatabucket', 'cub200data')
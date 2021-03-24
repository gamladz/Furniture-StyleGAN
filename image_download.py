import boto3
import os
import progressbar


def white_background():
    """
    Filter the downloaded images and take only those
    pictures with a white background
    """
    pass


def download(root='images', BUCKET_NAME='ikea-dataset'):

    # Check the size of the dataset
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    size = sum(1 for _ in bucket.objects.all())

    # Create a paginator object, so it ignores the 1000 limit
    client = boto3.client('s3')
    # Create a reusable Paginator
    paginator = client.get_paginator('list_objects')
    # Create a PageIterator from the Paginator
    # The Prefix='data' parameter ensures that we are only taking 
    # the images from the data folder
    page_iterator = paginator.paginate(Bucket=BUCKET_NAME,
                                       Prefix='data/')

    # Create a progress bar, so it tells how much is left
    print('Downloading...')
    bar = progressbar.ProgressBar(
            maxval=size,
            widgets=[progressbar.Bar('=', '[', ']'),
                    ' ', progressbar.Percentage()])
    bar.start()
    r = 0

    # Start the download
    for page in page_iterator:
        for content in page['Contents']:
            # Create a directory for each type of furniture ('bin', 'cookware'...)
            os.makedirs(f"{root}/{content['Key'].split('/')[1]}", exist_ok=True)
            LOCAL_FILE_NAME = (f"{root}/{content['Key'].split('/')[1]}"
                            + f"/{content['Key'].split('/')[-1]}")
            if not os.path.exists(LOCAL_FILE_NAME):
                client.download_file(BUCKET_NAME, content['Key'], LOCAL_FILE_NAME)
            
            # Update the progress bar
            bar.update(r + 1)
            r += 1
    bar.finish()
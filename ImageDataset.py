from PIL import Image
from PIL import ImageFile
import torch
import torchvision.transforms as transforms
import glob
import os
import platform
import boto3
import progressbar

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(torch.utils.data.Dataset):
    '''
    The ImageDataset object inherits its methods from the
    torch.utils.data.Dataset module.
    It loads all images from an image folder, creating labels
    for each image according to the subfolder containing the image
    and transformt the images so they can be used in a model
    Parameters
    ----------
    root_dir: str
        The directory with subfolders containing the images
    transform: torchvision.transforms 
        The transformation or list of transformations to be 
        done to the image. If no transform is passed,
        the class will do a generic transformation to
        resize, convert it to a tensor, and normalize the numbers
    
    Attributes
    ----------
    files: list
        List with the directory of all images
    labels: set
        Contains the label of each sample
    dict_encoder: dict
        Dictionary to translate the label to a 
        numeric value
    '''
    def __init__(self, root_dir, transform=None, download=True,
                 BUCKET_NAME='ikea-dataset'):
        
        self.root_dir = root_dir
        if download:
            self.download(self.root_dir, BUCKET_NAME)
        else:
            if not os.path.exists(root_dir):
                raise RuntimeError('Dataset not found.' +
                                   'You can use download=True to download it')
        # Check the OS of the machine
        if platform.system() == 'Darwin':
            self._sep = '/'
        else:
            self._sep = '\\'

        self.files = glob.glob(f'{self.root_dir}{self._sep}*{self._sep}*.jpg')
        self.labels = set([x.split(self._sep)[1] for x in self.files])
        self.num_classes = len(self.labels)
        self.dict_encoder = {y: x for (x, y) in enumerate(self.labels)}
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __getitem__(self, index):

        img_name = self.files[index]
        label = img_name.split(self._sep)[1]
        label = self.dict_encoder[label]
        label = torch.as_tensor(label)
        image = Image.open(img_name)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.files)
    
    def download(self,root, BUCKET_NAME):

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

def split_train_test(dataset, train_percentage):
    train_split = int(len(dataset) * train_percentage)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_split, len(dataset) - train_split]
    )
    return train_dataset, validation_dataset
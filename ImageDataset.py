import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
import requests
import random
import torch
import os
import random
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True


def repeat_channel(x):
            return x.repeat(3, 1, 1)

class ImageDataset(torch.utils.data.Dataset):
    '''
    The ImageDataset object inherits its methods from the
    torch.utils.data.Dataset module.
    It loads all images from an image folder, creating labels
    for each image according to the subfolder containing the image
    and transform the images so they can be used in a model
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
    encoder: dict
        Dictionary to translate the label to a 
        numeric value
    decoder: dict
        Dictionary to translate the numeric value
        to a label
    '''

    def __init__(self, root_dir, transform=None, download=False):
        
        self.root_dir = root_dir
        if download:
            self.download()
        else:
            if not os.path.exists(f'{self.root_dir}/images'):
                raise RuntimeError('Dataset not found.' +
                                   'You can use download=True to download it')

        with open(f'{self.root_dir}/chair_ikea.json') as json_file:
            data = json.load(json_file)

        files = []
        description = []
        price = []
        for furniture in data.keys():
            for product in data[furniture].keys():
                n_img = len(data[furniture][product]['image'])
                for image in data[furniture][product]['image']:
                    files.append(f"{self.root_dir}/images/{image.split('/')[1]}/{image.split('/')[-1]}")
                desc = [data[furniture][product]['description'].split(',')[0]]
                description.extend(desc * n_img)
                pri = [data[furniture][product]['price']]
                price.extend(pri * n_img)
        
        self.labels = description
        self.prices = price
        self.files = files
        self.num_classes = len(set(self.labels))
        self.encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.decoder = {x: y for (x, y) in enumerate(set(self.labels))}
        self.transform = transform
        if transform is True:
            self.transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) # is this right?
            ])

        self.transform_Gray = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Lambda(repeat_channel),
            transforms.Normalize((0.5,), (0.5,))
        ])


    def __getitem__(self, index):

        label = self.labels[index]
        label = self.encoder[label]
        label = torch.as_tensor(label)
        image = Image.open(self.files[index])
        if image.mode != 'RGB':
          image = self.transform_Gray(image)
        else:
          image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.files)
    
    def download(self):

        with open('DATA/chair_ikea.json') as f:
            data = json.load(f) # read data containing image paths

        paths = ('/'.join(path.split('/')[1:]) for category in data.values()
                 for item in category.values() for path in item['image']) # generate paths

        for path in tqdm(paths):
            # OS FRIENDLY WAYS TO GET THE IMG PATH AND DIR
            fp = os.path.join(self.root_dir, *os.path.split(path))
            if os.path.exists(fp):
                continue
            dir = os.path.join(*os.path.split(fp)[:1])
            Path(dir).mkdir(parents=True, exist_ok=True) # create dir if doesnt exist
            response = requests.get(f'https://ikea-dataset.s3.us-east-2.amazonaws.com/data/{path}')
            os.makedirs('DATA/images_test', exist_ok=True)
            fp_write = os.path.join('DATA/images', os.path.split(fp)[-1])
            with open(fp_write, 'wb') as f:
                f.write(response.content)

def split_train_test(dataset, train_percentage):
    train_split = int(len(dataset) * train_percentage)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_split, len(dataset) - train_split]
    )
    return train_dataset, validation_dataset


if __name__ == '__main__':
    dataset = ImageDataset(root_dir='images', download=True, transform=False)

    # GIVE RANDOM EXAMPLE
    while True:
        idx = random.randint(0, len(dataset))
        img, label = dataset[idx]
        print(label)
        img.show()
        break # your computer will die if you remove this
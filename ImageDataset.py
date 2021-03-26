from PIL import Image
from PIL import ImageFile
import torch
import torchvision.transforms as transforms
import os
import json
import requests
from pathlib import Path
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def tmp_func(x):
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
    dict_encoder: dict
        Dictionary to translate the label to a 
        numeric value
    '''
    def __init__(self, root_dir, transform=None, download=False,
                 BUCKET_NAME='ikea-dataset'):
        
        self.root_dir = root_dir
        if download:
            self.download(self.root_dir, BUCKET_NAME)
        else:
            if not os.path.exists(f'{self.root_dir}/images'):
                raise RuntimeError('Dataset not found.' +
                                   'You can use download=True to download it')
        # Keep this part of the code which organizes the data according to the 
        # folder containing the image
        
        # Check the OS of the machine
        # if platform.system() == 'Darwin':
        #     self._sep = '/'
        # else:
        #     self._sep = '\\' + '\\'
        # self._sep = '/'
        # self.files = glob.glob(f'{self.root_dir}{self._sep}*{self._sep}*.jpg')
        # self.labels = set([x.split(self._sep)[1] for x in self.files])
        
        # Instead we will use the json file with all the information about each piece 
        # of furniture
        
        with open(f'{self.root_dir}/chair_ikea.json') as json_file:
            data = json.load(json_file)

        files = []
        description = []
        price = []
        for furniture in data.keys():
            print(furniture)
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
        self.dict_encoder = {y: x for (x, y) in enumerate(set(self.labels))}
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        self.transform_Gray = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Lambda(tmp_func),
            transforms.Normalize((0.5,), (0.5,))
        ])


    def __getitem__(self, index):

        label = self.labels[index]
        label = self.dict_encoder[label]
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

        with open('DATA/data.json') as f:
            data = json.load(f) # read data containing image paths

        paths = ('/'.join(path.split('/')[1:]) for category in data.values()
                 for item in category.values() for path in item['images']) # generate paths

        for path in tqdm(paths):
            # OS FRIENDLY WAYS TO GET THE IMG PATH AND DIR
            fp = os.path.join(self.root_dir, *os.path.split(path))
            if os.path.exists(fp):
                continue
            dir = os.path.join(*os.path.split(fp)[:1])
            Path(dir).mkdir(parents=True, exist_ok=True) # create dir if doesnt exist
            response = requests.get(f'https://ikea-dataset.s3.us-east-2.amazonaws.com/data/{path}')
            with open(fp, 'wb') as f:
                f.write(response.content)
        

def split_train_test(dataset, train_percentage):
    train_split = int(len(dataset) * train_percentage)
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_split, len(dataset) - train_split]
    )
    return train_dataset, validation_dataset
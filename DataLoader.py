from PIL import Image
from PIL import ImageFile
import torch
import torchvision.transforms as transforms
import glob

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.files = glob.glob(f'{self.root_dir}/*.jpg')
        self.labels = set([x.split('\\')[1].split('\\')[0] for x in self.files])
        self.dict_encoder = {y: x for (x, y) in enumerate(self.labels)}
        self.transform = transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    # Not dependent on index
    def __getitem__(self, index):

        img_name = self.files[index]
        label = img_name.split('\\')[1].split('\\')[0]
        label = self.dict_encoder[label]
        label = torch.as_tensor(label)
        image = Image.open(img_name)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.files)
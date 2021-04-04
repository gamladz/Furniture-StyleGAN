import matplotlib.pyplot as plt
import torchvision
import torch
import argparse
import json
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def generate_image(n_imgs, label, dir, req_label):
    '''
    Creates a grid of fake n_imgs images
    that corresponds to the specified label
    Parameters
    ----------
    n_imgs: int
        Number of images to represent
    label: int
        Number corresponding to the label in the 
        encoder dictionary
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = torch.load(dir + 'Generator.pt', map_location=(device))
    noise = torch.randn(n_imgs, 100, 1, 1)
    if req_label:
        if n_imgs == 1:
            label = torch.as_tensor((label, ))
        else:
            label = [label] * n_imgs
            label = torch.as_tensor(label)
        fake = net(noise, label)
    else:
        fake = net(noise)
    grid_img = torchvision.utils.make_grid(fake, normalize=True)
    grid_img = grid_img.cpu().data.numpy()
    plt.imshow(grid_img.transpose(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            prog='Image Generation',
            description='Generates Furniture Images from a trained'
                         + 'generator model',
            allow_abbrev=True)
    parser.add_argument('--n_img', action='store', type=int,
                        help='number of images to generate',
                        default=32)
    parser.add_argument('--show', action='store_true',
                        help='list all possible labels',
                        default=False)
    parser.add_argument('--dir', action='store', type=str,
                        help='directory of the model',
                        default='toy_models/cWGAN/')
    parser.add_argument('--label', action='store', type=str,
                        help='label of the wanted images',
                        default='Chair')
    args = parser.parse_args()
    with open(args.dir + 'image_data.json') as f:
        image_data = json.load(f)
    
    if args.show:
        print('\n'.join(sorted(image_data['classes'])))
    else:
        if args.label in image_data['encoder']:
            label = image_data['encoder'][args.label]
            generate_image(args.n_img, label, args.dir,
                           image_data['requires_label'])
        else:
            print('Label not found. These are the available labels:')
            print(sorted(image_data['classes']))

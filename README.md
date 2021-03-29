# Furniture-StyleGAN

The project is part of the evaluation of Generative Adversarial Networks (GANs) for image synthesis. It uses a furniture dataset of ~1600 images to generate new images of chairs. The notebooks include implementations of DCGAN, CGAN and WGAN architectures. The main application is to allow furniture designers to generate new ideas and serve as an additional source of inspiration.

## Content

This repo has the following structure:

| Path | Description |
| ----- | ----- |
| Furniture-StyleGAN | Main directory |
| ├ Conditional_WGAN.ipynb | Notebook for training a cGAN with Weisserstein Loss |
| ├ DCGAN.ipynb | Notebook for training a Deep Convolutional GAN |
| ├ ImageDataset.py | Script which contains the class for defining the dataset |
| ├ WGAN.ipynb | Notebook for training a Deep Convolutional GAN with Wiesserstein Loss |
| ├ cGAN.ipynb | Notebook for training a Conditional GAN |
| └ Architectures | Folder that contains the architecture for each model |
|     ├ Conditional_WGAN_net.py | Notebook for training a cGAN with Weisserstein Loss |
| ├ DCGAN.ipynb | Notebook for training a Deep Convolutional GAN |
| ├ ImageDataset.py | Script which contains the class for defining the dataset |
| ├ WGAN.ipynb | Notebook for training a Deep Convolutional GAN with Wiesserstein Loss |
| ├ cGAN.ipynb | Notebook for training a Conditional GAN |


# Furniture-StyleGAN

The project is part of the evaluation of Generative Adversarial Networks (GANs) for image synthesis. It uses a furniture dataset of ~1600 images to generate new images of chairs. The notebooks include implementations of DCGAN, CGAN and WGAN architectures. The main application is to allow furniture designers to generate new ideas and serve as an additional source of inspiration.

![cWGAN_1000_epoch](https://user-images.githubusercontent.com/58112372/112843098-870fe680-90a2-11eb-98dc-60642ea4e235.gif)

## Content

This repo has the following structure:

| Path | Description |
| ----- | ----- |
| Furniture-StyleGAN | Main directory |
| ├ Conditional_WGAN.ipynb | Notebook for training a cGAN with Wasserstein Loss |
| ├ DCGAN.ipynb | Notebook for training a Deep Convolutional GAN. This notebook also includes the <br> hyperparameter tuning process that can be applied to any other network|
| ├ ImageDataset.py | Script which contains the class for defining the dataset. It includes the DataAugmentation, <br>how to deal with grayscale images, and it also returns the label of each image in case <br> we use Conditional GANs or Conditional WGANs |
| ├ ImageGeneration.py | Script that generates the images using the trained model. By default it uses the toy_models folder |
| ├ WGAN.ipynb | Notebook for training a Deep Convolutional GAN with Wasserstein Loss |
| ├ cGAN.ipynb | Notebook for training a Conditional GAN |
| ├ Architectures | Folder that contains the architecture for each model |
|  \| &nbsp;├ Conditional_WGAN_net.py | Script with the Discriminator and Generator architectures of the **Conditional WGAN** <br> network. It also contains the weights initialization and the gradient penalty functions|
|  \| &nbsp;├ DCGAN_net.py | Script with the Discriminator and Generator architectures of the **DCGAN** network <br> It also contains the weights initialization function.|
|  \| &nbsp;├ WGAN_net.ipy | Script with the Discriminator and Generator architectures of the **Conditional WGAN** network <br> It also contains the weights initialization and the gradient penalty functions. |
|  \| &nbsp;└ cGAN_net.py | Script with the Discriminator and Generator architectures of the **Conditional GAN** network <br> It also contains the weights initialization function. |
| └ DATA | Folder that contains the json files that includes the description of the furniture, as well as the link to download each image |
|  \| &nbsp;└ chair_ikea.json | JSON file with the url to download the images of the furniture. It also includes other characteristics, <br> such as the price, and the description (from which the label is extracted). <br>The current model is solely working with chairs, but more data will be added|
| └ toy_models | Folder that simulates the networks created after training the model in each notebook |
| &nbsp;&nbsp;└ Generator.pt | Pytorch model with the state_dicts of the trained generator network in the cWGAN model|
| &nbsp;&nbsp;└ Discriminator.pt | Pytorch model with the state_dicts of the trained discriminator network in the cWGAN model|
| &nbsp;&nbsp;└ image_data.json | json file with the information about the labels: num_classes, encoder and decoder|

## How to use this repo

The usage of this repo can be roughly divided into two: **Model training** and **Image Generation**

### Model Training

The name of each notebook corresponds to the model it will train. The notebooks have all the necessaries libraries included and will import the ImageDataset class, which takes images that are stored in an S3 AWS bucket if they haven't been downloaded yet. By default, the class will assume the data is already downloaded, so, in order to download the dataset, the class has to be initialized with a directory to save the images and the download argument set to True. <br>
An additional argument can be passed to the class: transform, which by default is True. If set to True, the images will be processed to work with any of the networks included in this repo. It will also transform the images to apply DataAugmentation. <br>

Once the dataset is loaded, the user can specify the parameters of the dataloader, such as batch size, number of workers, and shuffle. Changing the batch size migh affect the outcome of the training. <br>

A set of hyperparameters are initialized in the notebook. Learning rate, beta1, and batch size are used for tuning the hyperparameters, but other hyperparameters can be changed, such as beta2, or the gradient penalty multiplier if we are using Wasserstein loss function. <br>

Once the model runs, the output should look like this:

![Output_training](https://user-images.githubusercontent.com/58112372/112840650-e7515900-909f-11eb-8bea-e777d3da4ee4.png)

We will observe the loss function of the generator and discriminator (Loss_D, and Loss_G) and the probability distributions of the real and fake images (D(x), and D(G(z)))<br>
Simultaneously, the training loop stores these values and the generated image in a logs directory that can be accessed through tensorboard. Just note that the name of the logs of the conditional GAN and the DCGAN correspond to the values of their learning rate, beta1, and batch size, whereas the name of the logs of the cWGAN and WGAN are named after the variable they are tracking (lossD, lossG, penalty, fake image and real image).<br>
Thus, when running tensorboard after training, for example, the DCGAN by typing: <br>
```
%load_ext tensorboard
%tensorboard --logdir='logs_runs/logs_DCGAN'
```

Tensorboard should show something like this:

![DCGAN_scalars](https://user-images.githubusercontent.com/58112372/112842296-b3773300-90a1-11eb-94d6-e391490edf41.png)

If we go to the Images tab, we can see the fake generated images:

![DCGAN_images](https://user-images.githubusercontent.com/58112372/112842360-c1c54f00-90a1-11eb-80be-4509d18302f9.png)

If we run the Conditional WGAN, we will see the images throghout the training epochs: <br>
```
%load_ext tensorboard
%tensorboard --logdir='logs_runs/logs_cWGAN'
```


Once the model finishes training, it will be saved in a directory called model, so the user can use the generator to create new fake images with the already trained model, or can train them without the need of restarting from scratch.

### Image Generation

The created model can be used for generating images. In the DCGAN and the WGAN models, one can not specify the labels of the images to be generated. On the other hand, in the conditional models (cGAN and cWGAN), the label must be specified. <br>

To generate the images, the script to use is ImageGeneration.py, which accepts the following arguments: <br>
- n_img: number of images to generate
- show: show the possible labels
- label: furniture to generate
- dir: directory of the model containing the generator

In the command line, the user can write, for example:
```
python .\ImageGeneration.py --n_img 32 --dir toy_models/cWGAN/ --label 'Office chair'
```
And the outcome will be:

![32_Office_Chairs](https://user-images.githubusercontent.com/58112372/113498926-b0e96300-9511-11eb-88cb-e75847f0d5ef.png)



## Note

The code is meant to be used with any set of images, but the user must create a json file with the description and directory of the image. This will be changed to support other type of files or structures. 

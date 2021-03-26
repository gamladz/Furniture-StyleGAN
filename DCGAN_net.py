import torch.nn as nn

class Generator(nn.Module):
    '''
    Generator net for the DCGAN. It consists on
    Convolutional layers that increase the size of the 
    images while downsampling the number of channels.
    It takes random noises to generate images to
    make the Discriminator believe they are true, so the
    output has three channels and 64x64 size. 
    
    To change the output size, the user must take into
    account the next formula to obtain the desired size:
    
    Size_out ​= (Size_in​ − 1) x stride[0] − 2 × padding[0]
               + dilation[0] × (kernel_size[0] − 1)
               + output_padding[0] + 1
    Parameters
    ----------
    ngpu: int
        Number of gpus to use when parallelize processing 
        this network
    features_g: int
        Size of feature maps
    z_dim: int
        Size of the accepted latent vector (noise)
    img_ch: int
        Number of channels in the training images.
    Attributes
    ----------
    main: nn.Sequential
        Architecture of the network. In this case
        it consists on 5 Convolutional layers
    '''
    def __init__(self, ngpu, features_g, z_dim, img_ch):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            # state size. (features_g*8) x 4 x 4
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            # state size. (features_g*4) x 8 x 8
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            # state size. (features_g*2) x 16 x 16
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            # state size. (features_g) x 32 x 32
            nn.ConvTranspose2d(features_g, img_ch, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (img_ch) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    '''
    Discriminator net for the DCGAN. It consists on
    Convolutional layers that decrease the size of the 
    images while upsampling the number of channels.
    It takes real images and generated images by the 
    Generator and tells which images are real and 
    which ones are generated by the Generator.
    Therefore it accepts 64x64 images with three channels
    
    To change the input size, the user must take into
    account the next formula to obtain the desired size:
    
    Size_out​ = ([Size_in​ + 2 × padding[0] − 
                dilation[0] × (kernel_size[0]−1) − 1]
                / stride[0]) ​+1
    Parameters
    ----------
    ngpu: int
        Number of gpus to use when parallelize processing 
        this network
    features_d: int
        Size of feature maps
    img_ch: int
        Number of channels in the training images.
    Attributes
    ----------
    main: nn.Sequential
        Architecture of the network. In this case
        it consists on 5 Convolutional layers
    '''
    def __init__(self, ngpu, features_d, img_ch):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (img_ch) x 64 x 64
            nn.Conv2d(img_ch, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d) x 32 x 32
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*2) x 16 x 16
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*4) x 8 x 8
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (features_d*8) x 4 x 4
            nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


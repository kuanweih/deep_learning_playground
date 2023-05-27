import torch
import torch.nn as nn


if __name__ == "__main__":

    image_size = 224
    patch_size = 16
    num_layers = 12
    num_heads = 12
    hidden_dim = 768
    mlp_dim = 3072


    # fake input data
    n = 2  # batch size
    c = 3  # RGB
    h = image_size  # img size
    w = image_size  # img size

    x = torch.rand(n, c, h, w)
    print(x.shape)


    conv_proj = nn.Conv2d(
        in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
    
    x = conv_proj(x)
    print(x.shape)




# https://medium.com/analytics-vidhya/implementing-convolution-without-for-loops-in-numpy-ce111322a7cd
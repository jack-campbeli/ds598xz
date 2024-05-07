
""" Full assembly of the parts to form the complete network """

# from unet_parts import *


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes,global_length,bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.global_length = global_length

#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         factor = 2 if bilinear else 1
#         self.down3 = Down(256, 512 // factor)

#         # adding cbam during the downsampling of the unet
#         self.cbam1 = CBAM(channel=64)
#         self.cbam2 = CBAM(channel=128)
#         self.cbam3 = CBAM(channel=256)

#         self.up2 = Up(512+global_length, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x, x_global):
#         x1 = self.inc(x)    #output : 32*32*64
#         x1 = self.cbam1(x1) + x1

#         x2 = self.down1(x1)     #output : 16*16*128
#         x2 = self.cbam2(x2) + x2

#         x3 = self.down2(x2)     #output : 8*8*256
#         x3 = self.cbam3(x3) + x3

#         x4 = self.down3(x3)     #output : 4*4*512

#         x4 = torch.cat([x4, x_global], dim=1)

#         x = self.up2(x4, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

###########################################################################

import torch
from torch import nn
import torch.nn.functional as F
from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, global_length, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.global_length = global_length

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)  # Adding deeper layer

        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)
        self.cbam4 = CBAM(channel=512)  # New CBAM for the new layer

        # factor = 2 if bilinear else 1
        # self.up1 = Up(1024 + global_length, 512 // factor, bilinear)

        factor = 2 if bilinear else 1
        self.up1 = Up(1024 + 512 + 15, 512 // factor, bilinear)

        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_global):
        x1 = self.inc(x)
        x1 = self.cbam1(x1) + x1

        x2 = self.down1(x1)
        x2 = self.cbam2(x2) + x2

        x3 = self.down2(x2)
        x3 = self.cbam3(x3) + x3

        x4 = self.down3(x3)
        x4 = self.cbam4(x4) + x4

        #print(x4.shape)

        x5 = self.down4(x4)

        # print("x5 shape:", x5.shape)

        # zero padding  [ pad=(left,right,top,bottom) ]
        x5_new = F.pad(input=x5, pad=(1, 1, 1, 1), mode='constant', value=0)

        # print("x5_new shape:", x5_new.shape)
        # print("x_global shape:", x_global.shape)


        x5 = torch.cat([x5_new, x_global], dim=1)  # Adding global features at the lowest level

        # print("Concatenated x5 shape:", x5.shape)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

###########################################################################

# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, global_length, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.global_length = global_length

#         # Initialize standard UNet parts
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         factor = 2 if bilinear else 1
#         self.down3 = Down(256, 512 // factor)

#         # CBAM modules for enhanced feature attention
#         self.cbam1 = CBAM(channel=64)
#         self.cbam2 = CBAM(channel=128)
#         self.cbam3 = CBAM(channel=256)

#         # Enhance global features before concatenation
#         self.enhance_global = nn.Sequential(
#             nn.Conv2d(global_length, global_length, 3, padding=1),  # Additional feature enhancement
#             nn.ReLU(),
#             nn.Conv2d(global_length, 512, 1)  # Adjust channels to match x4
#         )

#         # Up-sampling layers adjusted for the combined features
#         self.up2 = Up(1024, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x, x_global):
#         x1 = self.inc(x)  # Initial double convolution
#         x1 = self.cbam1(x1) + x1  # Apply CBAM and residual connection

#         x2 = self.down1(x1)  # Downsample
#         x2 = self.cbam2(x2) + x2  # Apply CBAM and residual connection

#         x3 = self.down2(x2)  # Further downsample
#         x3 = self.cbam3(x3) + x3  # Apply CBAM and residual connection

#         x4 = self.down3(x3)  # Downsample to the lowest level

#         # Process the global features to enhance and match dimensions
#         x_global_enhanced = self.enhance_global(x_global)

#         # Concatenate the enhanced global features with the local features from x4
#         x4 = torch.cat([x4, x_global_enhanced], dim=1)

#         # Sequential up-sampling and combination with previous layers' outputs
#         x = self.up2(x4, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
        
#         # Final convolution to produce output predictions
#         logits = self.outc(x)
#         return logits

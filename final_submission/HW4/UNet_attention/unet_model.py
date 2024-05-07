
""" Full assembly of the parts to form the complete network """

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,global_length,bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.global_length = global_length

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)

        # adding cbam during the downsampling of the unet
        self.cbam1 = CBAM(channel=64)
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=256)

        self.up2 = Up(512+global_length, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, x_global):
        x1 = self.inc(x)    #output : 32*32*64
        x1 = self.cbam1(x1) + x1

        x2 = self.down1(x1)     #output : 16*16*128
        x2 = self.cbam2(x2) + x2

        x3 = self.down2(x2)     #output : 8*8*256
        x3 = self.cbam3(x3) + x3

        x4 = self.down3(x3)     #output : 4*4*512

        x4 = torch.cat([x4, x_global], dim=1)

        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

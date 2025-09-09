import torch
import torch.nn as nn

class DWConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class LiteUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=19, base=64):
        super().__init__()
        C = [in_ch, base, base*2, base*4, base*8]

        self.enc1 = DWConvBlock(C[0], C[1])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DWConvBlock(C[1], C[2])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DWConvBlock(C[2], C[3])
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DWConvBlock(C[3], C[4])

        self.up3 = nn.ConvTranspose2d(C[4], C[3], 2, stride=2)
        self.dec3 = DWConvBlock(C[3] + C[3], C[3])

        self.up2 = nn.ConvTranspose2d(C[3], C[2], 2, stride=2)
        self.dec2 = DWConvBlock(C[2] + C[2], C[2])

        self.up1 = nn.ConvTranspose2d(C[2], C[1], 2, stride=2)
        self.dec1 = DWConvBlock(C[1] + C[1], C[1])

        self.head = nn.Conv2d(C[1], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        x  = self.pool1(e1)
        e2 = self.enc2(x)
        x  = self.pool2(e2)
        e3 = self.enc3(x)
        x  = self.pool3(e3)

        x  = self.bottleneck(x)
        x  = self.up3(x)
        x  = torch.cat([x, e3], 1)
        x  = self.dec3(x)

        x  = self.up2(x)
        x  = torch.cat([x, e2], 1)
        x  = self.dec2(x)

        x  = self.up1(x)
        x  = torch.cat([x, e1], 1)
        x  = self.dec1(x)

        return self.head(x)
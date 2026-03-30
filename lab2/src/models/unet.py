import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),  # no padding
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3), # no padding
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def center_crop(feature_map, target_h, target_w):
    _, _, h, w = feature_map.size()
    start_y = (h - target_h) // 2
    start_x = (w - target_w) // 2
    return feature_map[:, :, start_y:start_y + target_h, start_x:start_x + target_w]


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.upconv4(b)
        e4_crop = center_crop(e4, d4.size(2), d4.size(3))
        d4 = torch.cat([e4_crop, d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        e3_crop = center_crop(e3, d3.size(2), d3.size(3))
        d3 = torch.cat([e3_crop, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        e2_crop = center_crop(e2, d2.size(2), d2.size(3))
        d2 = torch.cat([e2_crop, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        e1_crop = center_crop(e1, d1.size(2), d1.size(3))
        d1 = torch.cat([e1_crop, d1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)
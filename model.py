import torch
import torch.nn as nn

class EGAN_G(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        # self.avgpool = nn.AvgPool2d(8, stride=8)
        # self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.enc_conv1 = nn.Conv2d(1, 16, 5, stride=2, padding=2)
        self.enc_conv2 = nn.Conv2d(16, 64, 5, stride=2, padding=2)
        self.enc_conv3 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.enc_conv4 = nn.Conv2d(128, 512, 5, stride=2, padding=2)
        self.enc_conv5 = nn.Conv2d(512, 1024, 5, stride=2, padding=2)

        self.dec_conv1 = nn.Conv2d(1024, 2048, 3, stride=1, padding=1)
        self.dec_conv2 = nn.Conv2d(1024, 512, 3, stride=1, padding=1)
        self.dec_conv3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.dec_conv4 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.dec_conv5 = nn.Conv2d(32, 4, 3, stride=1, padding=1)

    def forward(self, x):
        # Downsample
        # x = self.avgpool(x)
        # Encoder
        enc1 = self.relu(self.enc_conv1(x))
        enc2 = self.relu(self.enc_conv2(enc1))
        enc3 = self.relu(self.enc_conv3(enc2))
        enc4 = self.relu(self.enc_conv4(enc3))
        enc5 = self.relu(self.enc_conv5(enc4))
        # Decoder
        x = self.relu(self.dec_conv1(enc5))
        x = self.pixel_shuffle(x)
        x = torch.cat((x, enc4), dim=1)
        x = self.relu(self.dec_conv2(x))
        x = self.pixel_shuffle(x)
        x = torch.cat((x, enc3), dim=1)
        x = self.relu(self.dec_conv3(x))
        x = self.pixel_shuffle(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.relu(self.dec_conv4(x))
        x = self.pixel_shuffle(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.relu(self.dec_conv5(x))
        x = self.pixel_shuffle(x)
        # Upsample
        # x = self.upsample(x)
        return x


class EGAN_D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers = []
        input_channel = 2
        width = 256

        # layers.append(nn.AvgPool2d(8, stride=8))

        for c in [64, 128]:
            layers.append(nn.Conv2d(input_channel, c, 3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Conv2d(c, c, 3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
            input_channel = c
            width //= 2

        for c in [256, 512, 512]:
            layers.append(nn.Conv2d(input_channel, c, 3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Conv2d(c, c, 3, stride=1, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
            layers.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
            input_channel = c
            width //= 2

        layers.append(nn.Flatten())
        layers.append(nn.Linear(width * width * input_channel, 2048))
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Linear(2048, 512))
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Linear(512, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

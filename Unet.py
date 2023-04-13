# import torch
# from torch import nn
# from torch.nn import functional as F
#
# class Conv_Block(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(Conv_Block, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(in_channel, out_channel, 3,1,1,padding_mode='reflect', bias=False),
#             nn.BatchNorm2d(out_channel),
#             nn.Dropout2d(0.3),
#             nn.LeakyReLU()
#         )
#
#     def forward(self, x):
#         return self.layer(x)
#
# class DownSample(nn.Module):
#     def __init__(self, channel):
#         super(DownSample, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(channel, channel ,3 ,2, 1, padding_mode='reflect', bias=False),
#             nn.BatchNorm2d(channel),
#             nn.LeakyReLU()
#         )
#     def forward(self,x):
#         return self.layer(x)
#
# class UpSample(nn.Module):
#     def __init__(self, channel):
#         super(UpSample, self).__init__()
#         self.layer = nn.Conv2d(channel, channel//2, 1, 1)
#
#     def forward(self, x, feature_map):
#         up = F.interpolate(x, scale_factor=2, mode='nearest')
#         out = self.layer(up)
#         return torch.cat((out, feature_map), dim=1)
# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()
#         self.c1 = Conv_Block(3, 64)
#         self.d1 = DownSample(64)
#         self.c2 = Conv_Block(64, 128)
#         self.d2 = DownSample(128)
#         self.c3 = Conv_Block(128, 256)
#         self.d3 = DownSample(256)
#         self.c4 = Conv_Block(256, 512)
#         self.d4 = DownSample(512)
#
#         self.c5 = Conv_Block(512, 1024)
#         self.u1 = UpSample(1024)
#         self.c6 = Conv_Block(1024, 512)
#         self.u2 = UpSample(512)
#         self.c7 = Conv_Block(512, 256)
#         self.u3 = UpSample(256)
#         self.c8 = Conv_Block(256, 128)
#         self.u4 = UpSample(128)
#         self.c9 = Conv_Block(128, 64)
#         self.out = nn.Conv2d(64, 3, 3, 1, 1)
#         self.Th = nn.Sigmoid()
#
#     def forward(self,x):
#         R1 = self.c1(x)
#         R2 = self.c2(self.d1(R1))
#         R3 = self.c3(self.d2(R2))
#         R4 = self.c4(self.d3(R3))
#         R5 = self.c5(self.d4(R4))
#
#         O1 = self.c6(self.u1(R5, R4))
#         O2 = self.c7(self.u2(O1, R3))
#         O3 = self.c8(self.u3(O2, R2))
#         O4 = self.c9(self.u4(O3, R1))
#
#         return self.Th(self.out(O4))
#
# if __name__ == "__main__":
#     x = torch.rand(2, 3, 256, 256)
#     net = UNet()
#     print(net(x).shape)
import torch
from torch import nn
from torchvision import models


def conv3x3(in_: int, out: int) -> nn.Module:
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_: int, out: int) -> None:
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, middle_channels: int, out_channels: int
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_filters: int = 32, pretrained: bool = False) -> None:
        """
        Args:
            num_filters:
            pretrained:
                False - no pre-trained network is used
                True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(
            num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8
        )
        self.dec5 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8
        )
        self.dec4 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4
        )
        self.dec3 = DecoderBlock(
            num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2
        )
        self.dec2 = DecoderBlock(
            num_filters * (4 + 2), num_filters * 2 * 2, num_filters
        )
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


# class Interpolate(nn.Module):
#     def __init__(
#         self,
#         size: int = None,
#         scale_factor: int = None,
#         mode: str = "nearest",
#         align_corners: bool = False,
#     ):
#         super().__init__()
#         self.interp = nn.functional.interpolate
#         self.size = size
#         self.mode = mode
#         self.scale_factor = scale_factor
#         self.align_corners = align_corners
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.interp(
#             x,
#             size=self.size,
#             scale_factor=self.scale_factor,
#             mode=self.mode,
#             align_corners=self.align_corners,
#         )
#         return x


# class DecoderBlockV2(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         middle_channels: int,
#         out_channels: int,
#         is_deconv: bool = True,
#     ):
#         super().__init__()
#         self.in_channels = in_channels
#
#         if is_deconv:
#             """
#                 Paramaters for Deconvolution were chosen to avoid artifacts, following
#                 link https://distill.pub/2016/deconv-checkerboard/
#             """
#
#             self.block = nn.Sequential(
#                 ConvRelu(in_channels, middle_channels),
#                 nn.ConvTranspose2d(
#                     middle_channels, out_channels, kernel_size=4, stride=2, padding=1
#                 ),
#                 nn.ReLU(inplace=True),
#             )
#         else:
#             self.block = nn.Sequential(
#                 Interpolate(scale_factor=2, mode="bilinear"),
#                 ConvRelu(in_channels, middle_channels),
#                 ConvRelu(middle_channels, out_channels),
#             )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.block(x)


class UNet16(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        num_filters: int = 32,
        pretrained: bool = False,
        is_deconv: bool = False,
    ):
        """
        Args:
            num_classes:
            num_filters:
            pretrained:
                False - no pre-trained network used
                True - encoder pre-trained with VGG16
            is_deconv:
                False: bilinear interpolation is used in decoder
                True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            self.encoder[0], self.relu, self.encoder[2], self.relu
        )

        self.conv2 = nn.Sequential(
            self.encoder[5], self.relu, self.encoder[7], self.relu
        )

        self.conv3 = nn.Sequential(
            self.encoder[10],
            self.relu,
            self.encoder[12],
            self.relu,
            self.encoder[14],
            self.relu,
        )

        self.conv4 = nn.Sequential(
            self.encoder[17],
            self.relu,
            self.encoder[19],
            self.relu,
            self.encoder[21],
            self.relu,
        )

        self.conv5 = nn.Sequential(
            self.encoder[24],
            self.relu,
            self.encoder[26],
            self.relu,
            self.encoder[28],
            self.relu,
        )

        self.center = DecoderBlockV2(
            512, num_filters * 8 * 2, num_filters * 8, is_deconv
        )

        self.dec5 = DecoderBlockV2(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv
        )
        self.dec4 = DecoderBlockV2(
            512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv
        )
        self.dec3 = DecoderBlockV2(
            256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv
        )
        self.dec2 = DecoderBlockV2(
            128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv
        )
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)
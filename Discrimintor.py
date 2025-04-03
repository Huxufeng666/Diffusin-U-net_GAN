import torch
import torch.nn as nn


import torch.nn.utils.spectral_norm as spectral_norm


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # 第一层：卷积 + LeakyReLU
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            # 第二层：卷积 + 批量归一化 + LeakyReLU
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 第三层：卷积 + 批量归一化 + LeakyReLU
            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 第四层：卷积 + 批量归一化 + LeakyReLU
            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出层：1x1 卷积，预测真假（0=假，1=真）
            spectral_norm(nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)),           
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)



class ComplexDiscriminator_pro(nn.Module):
    def __init__(self, in_channels=1):
        super(ComplexDiscriminator_pro, self).__init__()

        self.model = nn.Sequential(
            # 输入：(B, 1, 256, 256)
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),  # 128x128
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),  # 64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),  # 32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),  # 16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)),  # 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)),  # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)),  # 输出尺寸：(B,1,1,1)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)  # 展平成 (B, 1) 便于后续损失计算

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



# class ComplexDiscriminator_pro(nn.Module):
#     def __init__(self, in_channels=1):
#         super(ComplexDiscriminator_pro, self).__init__()

#         self.model = nn.Sequential(
#             # 输入：(B, 1, 256, 256)
#             spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),  # 128x128
#             nn.LeakyReLU(0.2, inplace=True),

#             spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),  # 64x64
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),

#             spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),  # 32x32
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),

#             spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),  # 16x16
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),

#             spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)),  # 8x8
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),

#             spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)),  # 4x4
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),

#             spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)),  # 输出尺寸：(B,1,1,1)
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         return self.model(x)  # 展平成 (B, 1) 便于后续损失计算

class ComplexDiscriminator_pro(nn.Module):
    def __init__(self, in_channels=1):
        super(ComplexDiscriminator_pro, self).__init__()
        self.in_channels = in_channels

        # Sobel 卷积核（用于边缘提取）
        self.edge_filter = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        sobel_kernel = torch.tensor([[[-1, -2, -1],
                                      [0,  0,  0],
                                      [1,  2,  1]]], dtype=torch.float32)  # Sobel Y
        self.edge_filter.weight.data = sobel_kernel.unsqueeze(0).repeat(in_channels, 1, 1, 1)
        self.edge_filter.weight.requires_grad = False  # 不训练边缘提取器

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 应用边缘提取器（非可学习）
        x = self.edge_filter(x)
        return self.model(x)
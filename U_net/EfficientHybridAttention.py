import torch
import torch.nn as nn

class EfficientHybridAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(EfficientHybridAttention, self).__init__()
        
        # ✅ 通道注意力 (SE Block)
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # ✅ 空间注意力 (Depthwise Separable Convolution)
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.spatial_bn = nn.BatchNorm2d(in_channels)
        self.spatial_act = nn.Sigmoid()

        # ✅ 混合权重
        self.weight_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # ✅ 通道注意力
        b, c, h, w = x.size()
        channel_weight = self.channel_avg_pool(x)
        channel_weight = self.channel_fc(channel_weight)
        
        # ✅ 空间注意力
        spatial_weight = self.spatial_conv(x)
        spatial_weight = self.pointwise_conv(spatial_weight)
        spatial_weight = self.spatial_bn(spatial_weight)
        spatial_weight = self.spatial_act(spatial_weight)

        # ✅ 混合权重
        hybrid_weight = self.weight_fc(x)
        
        # ✅ 特征融合
        out = channel_weight * x + spatial_weight * x + hybrid_weight * x
        
        return out


# ✅ 添加通道对齐卷积
class AttentionUNetpus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bilinear=True):
        super(AttentionUNetpus, self).__init__()
        
        # ✅ 编码器
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            EfficientHybridAttention(64)
        )
        
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            EfficientHybridAttention(128)
        )
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            EfficientHybridAttention(256)
        )
        
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            EfficientHybridAttention(512)
        )
        
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            EfficientHybridAttention(1024)
        )

        # ✅ 解码器 (添加通道对齐卷积)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # ✅ 编码路径
        x1 = self.in_conv(x)    # 64
        x2 = self.down1(x1)     # 128
        x3 = self.down2(x2)     # 256
        x4 = self.down3(x3)     # 512
        x5 = self.down4(x4)     # 1024

        # ✅ 解码路径
        x = self.up1(x5)        # 1024 -> 512
        x = torch.cat([x, x4], dim=1)  # 512 + 512 = 1024
        x = self.conv1(x)       # 1024 -> 512

        x = self.up2(x)         # 512 -> 256
        x = torch.cat([x, x3], dim=1)  # 256 + 256 = 512
        x = self.conv2(x)       # 512 -> 256

        x = self.up3(x)         # 256 -> 128
        x = torch.cat([x, x2], dim=1)  # 128 + 128 = 256
        x = self.conv3(x)       # 256 -> 128

        x = self.up4(x)         # 128 -> 64
        x = torch.cat([x, x1], dim=1)  # 64 + 64 = 128
        x = self.conv4(x)       # 128 -> 64

        return self.out_conv(x)

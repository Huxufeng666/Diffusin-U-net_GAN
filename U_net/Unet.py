import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        # 这里给出一个简化版本的 U-Net
        self.enc_conv1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc_conv2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = self.conv_block(128, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = self.conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = self.conv_block(128, 64)
        
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        # 这里我们不在 out_conv 中直接加入激活函数，
        # 后续计算损失时（例如BCEWithLogitsLoss）更方便

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # Encoder
        c1 = self.enc_conv1(x)         # shape: (B,64,H,W)
        p1 = self.pool1(c1)            # shape: (B,64,H/2,W/2)
        c2 = self.enc_conv2(p1)        # (B,128,H/2,W/2)
        p2 = self.pool2(c2)            # (B,128,H/4,W/4)
        # Bottleneck
        bn = self.bottleneck(p2)       # (B,256,H/4,W/4)
        # Decoder
        u2 = self.up2(bn)              # (B,128,H/2,W/2)
        # 拼接 encoder 层 c2
        u2_cat = torch.cat([u2, c2], dim=1)  # (B,256,H/2,W/2)
        d2 = self.dec_conv2(u2_cat)     # (B,128,H/2,W/2)
        u1 = self.up1(d2)              # (B,64,H,W)
        u1_cat = torch.cat([u1, c1], dim=1)  # (B,128,H,W)
        d1 = self.dec_conv1(u1_cat)     # (B,64,H,W)
        out = self.out_conv(d1)         # (B,1,H,W)
        return out





class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class UNets(nn.Module,):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNets, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits





# --------------------- Attention Block ---------------------
class A_SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(A_SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --------------------- Residual Block ---------------------
class A_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(A_ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))

# --------------------- Down, Up, Out ---------------------
class A_Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(A_Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = A_ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(self.pool(x))

class A_Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(A_Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 2, stride=2)

        self.conv = A_ResidualBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Padding in case input is not perfectly divisible
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# --------------------- Full Model ---------------------
class AttentionResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bilinear=True):
        super().__init__()
        self.in_conv = A_ResidualBlock(in_channels, 64)
        self.down1 = A_Down(64, 128)
        self.down2 = A_Down(128, 256)
        self.down3 = A_Down(256, 512)
        self.down4 = A_Down(512, 1024)

        self.se = A_SEBlock(1024)

        self.up1 = A_Up(1024 + 512, 512, bilinear)
        self.up2 = A_Up(512 + 256, 256, bilinear)
        self.up3 = A_Up(256 + 128, 128, bilinear)
        self.up4 = A_Up(128 + 64, 64, bilinear)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.se(x5)  # Attention module

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out_conv(x)
    
    
    
    
# ----------------------------------------------



class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


class UNetb(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetb, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Add residual blocks between upsampling steps
        self.block1 = ResBlock(512 // factor)
        self.block2 = ResBlock(256 // factor)
        self.block3 = ResBlock(128 // factor)
        self.block4 = ResBlock(64)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)      # 64
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024

        x = self.up1(x5, x4)
        x = self.block1(x)    # <- Add residual block after up1

        x = self.up2(x, x3)
        x = self.block2(x)

        x = self.up3(x, x2)
        x = self.block3(x)

        x = self.up4(x, x1)
        x = self.block4(x)

        logits = self.outc(x)
        return logits



# --------------------------------------------------------------------------------------------------




class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class NestedConvBlock(nn.Module):
    """U-Net++ Nested Block"""

    def __init__(self, in_channels, out_channels, layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 逐层添加卷积块
        for i in range(layers):
            self.layers.append(DoubleConv(in_channels, out_channels))
            in_channels = out_channels  # 下一个块的输入通道数变为输出通道数

    def forward(self, *inputs):
        # 将所有输入拼接起来
        x = torch.cat(inputs, dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


class UNetPlusPlus(nn.Module):
    """U-Net++ with Nested Skip Connections"""

    def __init__(self, n_channels, n_classes, bilinear=False, deep_supervision=False):
        super(UNetPlusPlus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DoubleConv(512, 1024 // factor)

        # Nested Skip Connections
        self.up1_1 = NestedConvBlock(1024 + 512, 512 // factor)
        self.up2_1 = NestedConvBlock(512 + 256, 256 // factor)
        self.up3_1 = NestedConvBlock(256 + 128, 128 // factor)
        self.up4_1 = NestedConvBlock(128 + 64, 64)

        # Second level nested blocks
        self.up2_2 = NestedConvBlock(512 + 256 + 256 // factor, 256 // factor)
        self.up3_2 = NestedConvBlock(256 + 128 + 128 // factor, 128 // factor)
        self.up4_2 = NestedConvBlock(128 + 64 + 64, 64)

        # Third level nested blocks
        self.up3_3 = NestedConvBlock(256 + 128 + 128 // factor + 128 // factor, 128 // factor)
        self.up4_3 = NestedConvBlock(128 + 64 + 64 + 64, 64)

        # Fourth level nested blocks
        self.up4_4 = NestedConvBlock(128 + 64 + 64 + 64 + 64, 64)

        # Final output
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0_0 = self.inc(x)
        x1_0 = self.down1(F.max_pool2d(x0_0, 2))
        x2_0 = self.down2(F.max_pool2d(x1_0, 2))
        x3_0 = self.down3(F.max_pool2d(x2_0, 2))
        x4_0 = self.down4(F.max_pool2d(x3_0, 2))

        # Nested paths
        x0_1 = self.up4_1(x0_0, F.interpolate(x1_0, scale_factor=2, mode='bilinear', align_corners=True))
        x1_1 = self.up3_1(x1_0, F.interpolate(x2_0, scale_factor=2, mode='bilinear', align_corners=True))
        x2_1 = self.up2_1(x2_0, F.interpolate(x3_0, scale_factor=2, mode='bilinear', align_corners=True))
        x3_1 = self.up1_1(x3_0, F.interpolate(x4_0, scale_factor=2, mode='bilinear', align_corners=True))

        # Second level
        x0_2 = self.up4_2(x0_0, x0_1, F.interpolate(x1_1, scale_factor=2, mode='bilinear', align_corners=True))
        x1_2 = self.up3_2(x1_0, x1_1, F.interpolate(x2_1, scale_factor=2, mode='bilinear', align_corners=True))
        x2_2 = self.up2_2(x2_0, x2_1, F.interpolate(x3_1, scale_factor=2, mode='bilinear', align_corners=True))

        # Third level
        x0_3 = self.up4_3(x0_0, x0_1, x0_2, F.interpolate(x1_2, scale_factor=2, mode='bilinear', align_corners=True))
        x1_3 = self.up3_3(x1_0, x1_1, x1_2, F.interpolate(x2_2, scale_factor=2, mode='bilinear', align_corners=True))

        # Fourth level
        x0_4 = self.up4_4(x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, scale_factor=2, mode='bilinear', align_corners=True))

        # 输出
        if self.deep_supervision:
            return [self.outc(x0_1), self.outc(x0_2), self.outc(x0_3), self.outc(x0_4)]
        else:
            return self.outc(x0_4)
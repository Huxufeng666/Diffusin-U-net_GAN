
import torch.nn as nn
from denoising_diffusion.Diffusion import  ContinuousTimeGaussianDiffusion,DiffusionModule

from denoising_diffusion import GaussianDiffusion
from UNet import UNet,UNets
import torch.nn.functional as F
# #############################
# # 4. 端到端模型
# #############################
import torch
import torch.nn.functional as F
from PIL import Image
import os

# 定义一个基本的残差块（Residual Block）
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入输出通道不匹配，需要使用1x1卷积调整输入通道的大小
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接
        out = self.relu(out)
        return out


# 定义ResNet网络
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet, self).__init__()
        
        self.channels = 1
        self.random_or_learned_sinusoidal_cond = False
        self.self_condition = False
        
        self.out_dim =    num_classes
        
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(self.channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 堆叠多个残差块
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 最终的全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, self.out_dim)
        self.out_conv =nn.Conv2d(512, self.out_dim, kernel_size=1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
 
    def forward(self, x,timestep=None, x_self_cond=None):
        
        if timestep is not None:
            pass
        if x_self_cond is not None:
            pass
        
        
        x = self.relu(self.bn1(self.conv1(x)))  # 初始卷积层
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.fc(x)
        x = F.interpolate(x, size=256, mode='bilinear', align_corners=False)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        
        x = self.out_conv(x)

        return x
    
   

class EndToEndModel(nn.Module):
    def __init__(self):
        super(EndToEndModel, self).__init__()
        # self.diffusion = ContinuousTimeGaussianDiffusion(kernel_size=5, sigma=1.0)
        # self.diffusion = ContinuousTimeGaussianDiffusion(
        #                                                 model=ResNet(BasicBlock, [2, 2, 2, 2]) ,
        #                                                 image_size=256,
        #                                                 channels=1,
        #                                                 noise_schedule='linear',
        #                                                 num_sample_steps=10000,
        #                                                 clip_sample_denoised=True,
        #                                                 learned_schedule_net_hidden_dim=1024,
        #                                                 learned_noise_schedule_frac_gradient=1.,
        #                                                 min_snr_loss_weight=False,
        #                                                 min_snr_gamma=5,
        #                                                 kernel_size=5,
        #                                                 sigma=1.0
        #                                             )
        
        self.diffusion = GaussianDiffusion(
                                    model= ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1),
                                    image_size=256,                # 图像尺寸为256x256
                                    timesteps=1000,                # 总的扩散时间步数
                                    sampling_timesteps=None,       # 采样时步数，默认为训练时步数
                                    objective='pred_noise',        # 训练目标，这里选择预测噪声
                                    beta_schedule='sigmoid',       # 使用 sigmoid 的 beta 调度
                                    schedule_fn_kwargs={},         # 可选参数
                                    ddim_sampling_eta=0.0,         # DDIM采样参数
                                    auto_normalize=True,           # 自动归一化数据
                                    offset_noise_strength=0.0,     # 偏移噪声强度
                                    min_snr_loss_weight=False,     # 是否采用最小信噪比损失权重
                                    min_snr_gamma=5,
                                    immiscible=False               # 是否采用不混合扩散
                                )
                                        
        # self.unet = UNets(in_channels=1, out_channels=1)
        self.unet = UNets(n_channels=1, n_classes=1)
    
    def forward(self, image, mask):
        # Diffusion 模块：对图像进行背景模糊（突出肿瘤区域）
        processed_image = self.diffusion(image, mask,)
        
        # proc_img = processed_image[0].cpu().detach()  # shape: [1, H, W]
        # proc_img = proc_img.squeeze(0)                # shape: [H, W]

        # # 如果图像值在 [0, 1]，将其乘以 255 并转换为 uint8
        # proc_img_np = (proc_img.numpy() * 255).astype('uint8')

        # # 将 NumPy 数组转换为 PIL 图像
        # proc_pil = Image.fromarray(proc_img_np)

        # # 定义保存路径，确保文件夹存在
        # save_dir = "processed_images"
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # save_path = os.path.join(save_dir, "processed_image.png")
        # proc_pil.save(save_path)
        # print(f"Processed image saved to: {save_path}")
        # # U-Net 分割：对处理后的图像进行分割    if isinstance(result, tuple):
        
        if isinstance(processed_image,tuple):
            processed_image = processed_image[1]  # 只取图像部分
        else:
            processed_image = processed_image
        
        segmentation = self.unet(processed_image)
        return processed_image, segmentation

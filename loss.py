import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# BCE损失
class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        return self.bce_loss(inputs, targets)


# Dice损失
class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        smooth = self.epsilon
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)
        return 1 - (2. * intersection + smooth) / (union + smooth)
    
    

# SSIM损失函数
class SSIMLoss(nn.Module,):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.window = self.create_window(window_size)

    def create_window(self, window_size):
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        # 生成一个高斯窗用于计算SSIM
        window = torch.ones(1, 1, window_size, window_size)
        return window.to(device)

    def gaussian(self, window_size, sigma):
        # 生成高斯滤波器
        gauss = torch.Tensor([math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        return gauss

    def forward(self, img1, img2):
        # 计算SSIM损失
        mu1 = F.conv2d(img1, self.window, padding=3)
        mu2 = F.conv2d(img2, self.window, padding=3)
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=3) - mu1 * mu1
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=3) - mu2 * mu2
        sigma12 = F.conv2d(img1 * img2, self.window, padding=3) - mu1 * mu2

        C1 = 0.01**2
        C2 = 0.03**2

        # SSIM公式
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean() if self.size_average else 1 - ssim_map.mean(dim=1).mean(dim=1).mean(dim=1)



class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5):
        super().__init__()
        self.bce_loss = BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.ssim_loss = SSIMLoss()
        self.alpha = alpha  # 权重系数，用于调节BCE与Dice损失的平衡
        self.beta = beta  # 权重系数，用于调节SSIM损失的影响
        self.gamma = gamma  # 如果需要其他损失，进行调整

    def forward(self, outputs, targets):
        # 计算每个损失
        bce = self.bce_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        ssim = self.ssim_loss(outputs, targets)

        # 结合多个损失
        total_loss = self.alpha * bce + self.beta * dice + self.gamma * ssim
        return total_loss





def plot_losses(csv_path, output_path='loss_curve.png'):
    df = pd.read_csv(csv_path)

    # 可选：过滤异常高的 Test Seg Loss（如 > 5）
    df_filtered = df[df['Test Seg Loss'] < 5]

    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['generator_loss'], label='Generator Loss')
    plt.plot(df['epoch'], df['discriminator_loss'], label='Discriminator Loss')
    plt.plot(df_filtered['epoch'], df_filtered['Test Seg Loss'], label='Test Seg Loss (filtered)', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    plt.savefig(output_path)
    plt.close()
    
def plot_losses_2(csv_path, output_prefix='loss_curve'):
    # 读取 CSV 数据
    df = pd.read_csv(csv_path)
    
    # 可选：过滤异常高的 Test Seg Loss（如 > 5）
    df_filtered = df[df['Test Seg Loss'] < 5]
    
    # 1. 绘制 Generator Loss 曲线
    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['generator_loss'], label='Generator Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    generator_path = f"{output_prefix}_generator.png"
    plt.savefig(generator_path)
    plt.close()
    print(f"Generator loss curve saved to: {generator_path}")
    
    # 2. 绘制 Discriminator Loss 曲线
    plt.figure(figsize=(12, 6))
    plt.plot(df['epoch'], df['discriminator_loss'], label='Discriminator Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    discriminator_path = f"{output_prefix}_discriminator.png"
    plt.savefig(discriminator_path)
    plt.close()
    print(f"Discriminator loss curve saved to: {discriminator_path}")
    
    # 3. 绘制 Test Seg Loss 曲线
    plt.figure(figsize=(12, 6))
    plt.plot(df_filtered['epoch'], df_filtered['Test Seg Loss'], label='Test Seg Loss (filtered)', linestyle='--', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Seg Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    test_seg_path = f"{output_prefix}_test_seg.png"
    plt.savefig(test_seg_path)
    plt.close()
    print(f"Test segmentation loss curve saved to: {test_seg_path}")   


def compute_edge_from_mask(mask_tensor, threshold=0.5):
    """
    输入：mask_tensor，形状为 [batch, 1, H, W]（取值在 [0,1]）
    输出：边缘图，形状同 mask_tensor
    """
    mask_np = mask_tensor.squeeze(1).cpu().numpy()  # shape: [batch, H, W]
    edges = []
    for m in mask_np:
        # 将 mask 转为 uint8 类型
        m_uint8 = (m >= threshold).astype(np.uint8) * 255
        # 使用 Canny 边缘检测
        edge = cv2.Canny(m_uint8, 100, 200)
        edge = edge.astype(np.float32) / 255.0  # 归一化到 [0,1]
        edges.append(edge)
    edges = np.expand_dims(np.array(edges), 1)  # shape: [batch, 1, H, W]
    return torch.tensor(edges, dtype=torch.float32, device=mask_tensor.device)

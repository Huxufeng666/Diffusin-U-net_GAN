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
class DiceLoss_T(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon  # 修复点：使用 self.epsilon 而不是 self.smooth

    def forward(self, inputs, targets):
        # Flatten the tensors
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)
        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)

        return 1 - dice
    
    
    
class DiceLoss_v(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # 使用 torch 张量运算，不需要 .astype()（这是 numpy 的方法）
        inputs = torch.sigmoid(inputs)  # 如果未归一化要先 Sigmoid
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)

        return  dice  # 损失越小越好        
    

# SSIM损失函数
class SSIMLoss(nn.Module,):
    def __init__(self, window_size=7, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.window = self.create_window(window_size)
        self.window = self.window.expand(1, 1, self.window_size, self.window_size).contiguous()


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



class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight)
        self.smooth = smooth

    def forward(self, inputs, targets):
        # BCE Loss
        bce = self.bce_loss(inputs, targets)

        # Dice Loss
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        dice_loss = 1 - dice

        # BCE + Dice Loss
        total_loss = bce + dice_loss
        return total_loss




class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5):
        super().__init__()
        self.bce_loss = BCEWithLogitsLoss()
        self.dice_loss = DiceLoss_T()
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



def dice_coefficient(pred, target, smooth=1e-6):
    """
    计算两个二值图像之间的Dice系数。

    Args:
        pred (np.ndarray): 预测结果二值图 (numpy数组, shape: H x W 或 B x H x W)。
        target (np.ndarray): 真实掩码二值图 (numpy数组, shape: 同pred相同)。
        smooth (float): 防止分母为0的平滑项。

    Returns:
        float: Dice系数，范围[0, 1]。
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    intersection = np.sum(pred_flat * target_flat)
    dice = (2. * intersection + smooth) / (np.sum(pred_flat) + np.sum(target_flat) + smooth)

    return dice




def plot_losses(csv_file, save_path):
    df = pd.read_csv(csv_file)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    
    if 'Test Loss' in df.columns:
        plt.plot(df['Epoch'], df['Test Loss'], label='Test Loss')
    # if 'Dice' in df.columns:
    #     plt.plot(df['Epoch'], df['Dice'], label='Dice Coefficient', linestyle='--') 

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()



def  plot_losses_pros(csv_path, save_path):
    """
    从CSV文件中绘制训练损失、测试损失和Dice系数。

    Args:
        csv_path (str): CSV文件路径。
        save_path (str): 图像保存路径。
    """
    # 加载数据
    df = pd.read_csv(csv_path)

    # 创建大画布与子画布
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 绘制训练损失
    axes[0].plot(df['Epoch'], df['Train Loss'], color='blue', label='Train Loss')
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # 绘制测试损失（若存在）
    if 'Test Loss' in df.columns:
        axes[1].plot(df['Epoch'], df['Test Loss'], color='green', label='Test Loss')
        axes[1].set_title('Test Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].set_visible(False)

    # 绘制Dice系数（若存在）
    if 'Dice' in df.columns:
        axes[2].plot(df['Epoch'], df['Dice'], color='red', linestyle='--', label='Dice Coefficient')
        axes[2].set_title('Dice Coefficient')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Dice')
        axes[2].legend()
        axes[2].grid(True)
    else:
        axes[2].set_visible(False)

    # 整体标题与布局优化
    plt.suptitle('Training Metrics Overview', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存图像
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()





def plot_losses_2(csv_path, output_prefix='loss_curve'):
    df = pd.read_csv(csv_path)

    # 绘制 Generator Loss
    if 'generator_loss' in df.columns:
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

    # 绘制 Discriminator Loss
    if 'discriminator_loss' in df.columns:
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

    # 绘制 Test Segmentation Loss（如果存在）
    if 'Test Seg Loss' in df.columns:
        df_filtered = df[df['Test Seg Loss'] < 5]  # 可选过滤
        plt.figure(figsize=(12, 6))
        plt.plot(df_filtered['epoch'], df_filtered['Test Seg Loss'], label='Test Seg Loss (filtered)', linestyle='--', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Test Segmentation Loss Curve')
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision import models

# # VGG-based perceptual loss feature extractor
# class VGGPerceptual(nn.Module):
#     def __init__(self, layer_index=9):
#         super().__init__()
#         vgg = models.vgg16(pretrained=True).features[:layer_index]
#         for param in vgg.parameters():
#             param.requires_grad = False
#         self.vgg = vgg

#     def forward(self, x):
#         if x.shape[1] == 1:
#             x = x.repeat(1, 3, 1, 1)  # convert grayscale to 3-channel
#         return self.vgg(x)
    

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_index=9):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features[:layer_index]
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg.eval()

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        self.vgg = self.vgg.to(x.device)
        fx = self.vgg(x)
        fy = self.vgg(y)
        return F.l1_loss(fx, fy)

# Custom loss module for diffusion-style generation with diversity and structure constraints
class CustomDiffusionLoss(nn.Module):
    def __init__(self, lambda_diff=1.0, lambda_struct=0.5, lambda_diverse=0.3, lambda_gan=0.2):
        super().__init__()
        self.lambda_diff = lambda_diff
        self.lambda_struct = lambda_struct
        self.lambda_diverse = lambda_diverse
        self.lambda_gan = lambda_gan

        self.perceptual_net = VGGPerceptualLoss().to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).eval()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()

    def perceptual_loss(self, x, y):
        self.perceptual_net = self.perceptual_net.to(x.device)
        fx = self.perceptual_net(x)
        fy = self.perceptual_net(y)
        return self.l1(fx, fy)

    def forward(self, pred_noise, true_noise, gen_img, input_img, d_output_fake=None):
        # 1. diffusion base loss
        loss_diff = self.mse(pred_noise, true_noise)

        # 2. structure-preserving loss (similar to input)
        loss_struct = self.perceptual_loss(gen_img, input_img)

        # 3. diversity loss (make image NOT like input)
        loss_diverse = -loss_struct  # encourage difference

        # 4. GAN loss (optional, if discriminator used)
        if d_output_fake is not None:
            loss_gan = self.bce(d_output_fake, torch.ones_like(d_output_fake))
        else:
            loss_gan = torch.tensor(0.0, device=gen_img.device)

        # total
        total = (
            self.lambda_diff * loss_diff +
            self.lambda_struct * loss_struct +
            self.lambda_diverse * loss_diverse +
            self.lambda_gan * loss_gan
        )

        return total, {
            "loss_total": total.item(),
            "loss_diff": loss_diff.item(),
            "loss_struct": loss_struct.item(),
            "loss_diverse": loss_diverse.item(),
            "loss_gan": loss_gan.item() if isinstance(loss_gan, torch.Tensor) else 0.0
        }






 
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




def tversky_loss(pred, target, alpha=0.5, beta=0.5, smooth=1.0):
    TP = (pred * target).sum(dim=[2, 3])
    FP = (pred * (1 - target)).sum(dim=[2, 3])
    FN = ((1 - pred) * target).sum(dim=[2, 3])
    
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return 1 - tversky.mean()


# BCE损失
class BCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        return self.bce_loss(inputs, targets)


# Dice损失
class DiceLoss_T(nn.Module):
    def __init__(self, epsilon=1e-4):
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
    

 # SSIM损失函数
class SSIMLoss(nn.Module,):
    def __init__(self, window_size=7, size_average=True,data_range = 1.0):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        window = self.create_window(window_size)
        self.window = self.create_window(window_size)
        self.data_range = data_range
        self.window = self.window.expand(1, 1, self.window_size, self.window_size).contiguous()
        # self.register_buffer("window", window)  # ✅ 注册为 buffer


    def create_window(self, window_size):
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        # 生成一个高斯窗用于计算SSIM
        window = torch.ones(1, 1, window_size, window_size)
        return window.to(device)


    # def create_window(self, window_size, sigma=1.5):
    #     # 确保 window_size 是一个 int
    #     if isinstance(window_size, torch.Tensor):
    #         if window_size.numel() == 1:  # 只允许单元素 Tensor
    #             window_size = int(window_size.item())
    #         else:
    #             raise ValueError("window_size should be a single integer, not a tensor with multiple elements.")
        
    #     # 创建二维高斯窗口
    #     gauss = torch.tensor([
    #         math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) 
    #         for x in range(window_size)
    #     ])
    #     gauss = gauss / gauss.sum()

    #     # 生成二维高斯核
    #     window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)  # (window_size, window_size)
    #     window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, window_size, window_size)

    #     # 将 window 移动到 GPU 或 CPU
    #     return window.to(self.window.device if hasattr(self, 'window') else 'cpu')
    
    
    def gaussian(self, window_size, sigma):
        # 生成高斯滤波器
        gauss = torch.Tensor([math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        return gauss

    def forward(self, img1, img2):
        
        img1 = img1 / self.data_range
        img2 = img2 / self.data_range
        # 计算SSIM损失
        mu1 = F.conv2d(img1, self.window, padding=3)
        mu2 = F.conv2d(img2, self.window, padding=3)
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=3) - mu1 * mu1
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=3) - mu2 * mu2
        sigma12 = F.conv2d(img1 * img2, self.window, padding=3) - mu1 * mu2

        C1 = (0.01 ** 2) * self.data_range ** 2
        C2 = (0.03 ** 2) * self.data_range ** 2

        # SSIM公式
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean() if self.size_average else 1 - ssim_map.mean(dim=1).mean(dim=1).mean(dim=1)
 

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.5):
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
    
   
class CombinedLoss_pro(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5, delta=0.3):
        super().__init__()
        self.bce_loss = BCEWithLogitsLoss()
        self.dice_loss = DiceLoss_T()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta  # 新加的多样性损失权重

    def forward(self, outputs, targets, input_img=None):
        bce = self.bce_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets)
        ssim = self.ssim_loss(outputs, targets)

        # 感知差异损失（鼓励生成结果 ≠ 输入图）
        if input_img is not None:
            diversity = - self.perceptual_loss(outputs, input_img)  # 多样性损失
        else:
            diversity = 0.0

        total_loss = self.alpha * bce + self.beta * dice + self.gamma * ssim + self.delta * diversity
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
    
    if 'Val Loss' in df.columns:
        plt.plot(df['Epoch'], df['Val Loss'], label='Val Loss')
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
    if 'Val Loss' in df.columns:
        axes[1].plot(df['Epoch'], df['Val Loss'], color='green', label='Val Loss')
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

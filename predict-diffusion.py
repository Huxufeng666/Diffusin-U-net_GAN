import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型（你需要根据自己定义的类导入）
# from your_model_file import GaussianDiffusion, ResNet, BasicBlock  # 替换为你真实的路径和类名
from denoising_diffusion import GaussianDiffusion
from model import EndToEndModel, ResNet, BasicBlock

# 初始化模型
model = GaussianDiffusion(
    model=ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1),
    image_size=256,
    timesteps=1000,
    objective='pred_noise',
    beta_schedule='sigmoid',
    auto_normalize=True,
    offset_noise_strength=0.0,
    min_snr_loss_weight=False,
    min_snr_gamma=5,
    immiscible=False
).to(device)

# 加载预训练权重
model.load_state_dict(torch.load("weights/2025-04-14_13-10-59_Diffusion/diffusion/diffusion_epoch_13.pth", map_location=device))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载输入图像
img_path = "/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/BUSI/train/images/benign_benign (26).png"
image = Image.open(img_path).convert("L")
input_tensor = transform(image).unsqueeze(0).to(device)
mask = 0

# 推理
with torch.no_grad():
    # output = model.sample(batch_size=1)  # 或者 model(input_tensor, mask=None)[1]
    output = model(input_tensor, mask)[1]

# 显示预测结果
pred = output.squeeze().cpu().numpy()
save_path = '1.png'
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Input Image")

plt.subplot(1, 2, 2)
plt.imshow(pred, cmap='gray')
plt.title("Generated Output")
plt.tight_layout()
plt.show()
plt.savefig(save_path)
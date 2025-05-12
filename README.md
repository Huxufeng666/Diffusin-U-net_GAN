# Diffusion-U-Net-GAN for Ultrasound Tumor Segmentation
[📖 English Version → README_EN.md](README_EN.md)

本项目基于扩散模型（Diffusion Model）结合 U-Net 和 GAN 判别器，用于超声图像中的肿瘤去噪与自动分割，适用于医学图像处理、智能诊断系统等场景。

---

## 📌 项目功能

- ✅ 超声图像去噪（Diffusion）
- ✅ 肿瘤区域精准分割（U-Net）
- ✅ 真实/伪生成图像判别（GAN）
---

## 🧱 系统结构
 

系统流程图：

![System Architecture](image.png)

---

## 📁 项目结构

```
├── models/             # 模型定义（Diffusion, U-Net, Discriminator）
├── datasets/           # 数据加载与预处理
├── train.py            # 训练主文件
├── predict.py          # 推理与结果保存
├── utils/              # 工具函数（可视化、loss 计算）
├── requirements.txt    # Python依赖
└── README.md
```

---



---

## 📁 结果


<td rowspan="2"> 

| Dataset          | Model                 | Dice Coefficient | IoU    | Pixel Accuracy | Precision | Recall |
| ---------------- | --------------------- | ---------------- | ------ | -------------- | --------- | ------ |
|**BUSI data**     | **AttUNet**           | 76.14            | 71.25  | 95.60          | 80.25     | 80.17  |
|                  | **Diffusion-AttUNet** | 77.96            | 71.18  | 96.77          | 88.56     | 79.06  |
| **BUS data**     | **AttUNet**           | 78.93            | 69.51  | 96.44          | 82.43     | 80.25  |
|                  | **Diffusion-AttUNet** | 77.21            | 75.00  | 99.03          | 86.47     | 88.33  |
| **BUS-BRA data** | **AttUNet**           | 0.893            | 0.823  | 0.980          | 0.909     | 0.906  |
|                  | **Diffusion-AttUNet** | 0.8952           | 0.8136 | 0.975          | 0.911     | 0.880  |



---



## 🚀 快速开始

### 1️⃣ 安装环境

```bash
conda create -n diffgan python=3.8
conda activate diffgan
pip install -r requirements.txt
```

### 2️⃣ 训练模型（支持多种模式）

🧪 使用命令：

✅ 1. 单独训练 diffusion 模型
```bash
python train.py --train_mode diffusion_only
```

✅ 2. 单独训练 U-Net（不加载 diffusion 模块）
```bash
python train.py --train_mode unet_only
```
可选参数：加载预训练 U-Net：
```bash
--unet_ckpt weights/unet_only/unet_epoch_50.pth
```

✅ 3. 冻结 diffusion，训练 U-Net
```bash
python train.py --train_mode finetune_unet --diffusion_ckpt weights/diffusion/diffusion_epoch_100.pth
```

✅ 4. 先训练 diffusion，再自动训练 U-Net
```bash
python train.py --train_mode full_pipeline
```

### 3️⃣ 测试与分割展示

```bash
python predict.py --image sample.png
```

---

## 🙋‍♀️ 项目作者

- **胡旭峰 Huxufeng**
- 📧 Email: `hxufneg66@gmail.com`
- GitHub: [Huxufeng666](https://github.com/Huxufeng666)

---

## 📄 License

本项目遵循 MIT 开源协议，欢迎使用、改进与引用。








# Diffusion-U-Net-GAN for Ultrasound Tumor Segmentation

This project integrates a **Diffusion Model**, **U-Net**, and **GAN Discriminator** for ultrasound image denoising and automatic tumor segmentation. It is suitable for tasks in medical image analysis and intelligent diagnostic systems.

---

## 📌 Features

- ✅ Ultrasound image denoising (Diffusion)
- ✅ Precise tumor region segmentation (U-Net)
- ✅ Real/Fake image discrimination (GAN)
- ✅ Cloud-based deployment with real-time result display
- ✅ Support for environment data collection and alert notifications

---

## 🧱 System Architecture

System workflow:

![System Architecture](Architecture.png)

---

## 📁 Project Structure

```
├── models/             # Model definitions (Diffusion, U-Net, Discriminator)
├── datasets/           # Data loading and preprocessing
├── train.py            # Main training script
├── predict.py          # Inference and result saving
├── utils/              # Utility functions (visualization, loss calculation)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Set up Environment

```bash
conda create -n diffgan python=3.8
conda activate diffgan
pip install -r requirements.txt
```

### 2️⃣ Train the Model (Various Modes Supported)

🧪 Example commands:

✅ 1. Train diffusion model only
```bash
python train.py --train_mode diffusion_only
```

✅ 2. Train U-Net only (without diffusion)
```bash
python train.py --train_mode unet_only
```
Optional: load pretrained U-Net weights:
```bash
--unet_ckpt weights/unet_only/unet_epoch_50.pth
```

✅ 3. Freeze diffusion and train U-Net
```bash
python train.py --train_mode finetune_unet --diffusion_ckpt weights/diffusion/diffusion_epoch_100.pth
```

✅ 4. Full pipeline: train diffusion first, then train U-Net automatically
```bash
python train.py --train_mode full_pipeline
```

### 3️⃣ Run Inference

```bash
python predict.py --image sample.png
```

---

## 🙋‍♀️ Author

- **Huxufeng**
- 📧 Email: `hxufneg66@gmail.com`
- GitHub: [Huxufeng666](https://github.com/Huxufeng666)

---

## 📄 License

This project is licensed under the MIT License. Feel free to use, modify, and cite it.

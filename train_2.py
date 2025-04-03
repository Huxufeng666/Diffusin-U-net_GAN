import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data import MedicalDataset, UltrasoundDataset
import torch.optim as optim
import torch.nn as nn
from Discrimintor import Discriminator,ComplexDiscriminator_pro
from model import EndToEndModel, ResNet, BasicBlock
import os
import csv
import datetime
from tqdm import tqdm
from loss import CombinedLoss, compute_edge_from_mask, plot_losses, plot_losses_2
import torch.nn.functional as F
from monai.losses import HausdorffDTLoss
from denoising_diffusion import GaussianDiffusion
from UNet import UNet, UNets
from torchvision.utils import save_image
from configs.save_diffusion_comparison import save_diffusion_comparison
# from torchvision.utils import save_image

def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion + U-Net Model")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--data_path', type=str, default='/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/BUS-UCLM/partitions')
    parser.add_argument('--weights_dir', type=str, default='weights')
    parser.add_argument('--train_mode', type=str, default='finetune_unet', choices=['diffusion_only', 'finetune_unet', 'unet_only', 'full_pipeline'])
    parser.add_argument('--unet_ckpt', type=str, default="", help='Path to U-Net pretrained weights')
    parser.add_argument('--diffusion_ckpt', type=str, default='weights/2025-04-03 10:12:37/diffusion/diffusion_epoch_100.pth', help='Path to diffusion model checkpoint')
    return parser.parse_args()




def train_diffusion(args, device):
    
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    weights_dir = os.path.join(args.weights_dir, formatted_time)
    os.makedirs(weights_dir, exist_ok=True)
    
    
    csv_file = os.path.join(weights_dir, "training_diffusio_log.csv")

    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    train_dataset = UltrasoundDataset(
        image_dir=os.path.join(args.data_path, 'train', 'images'),
        mask_dir=os.path.join(args.data_path, 'train', 'masks'),
        transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    diffusion_model = GaussianDiffusion(
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

    discriminator = ComplexDiscriminator_pro().to(device)
    criterion_gan = CombinedLoss(alpha=1.0, beta=1.0, gamma=0.2)

    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=args.lr)
    
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    print("[INFO] Starting diffusion training...")
    for epoch in range(100):  # 默认训练100个 epoch
        diffusion_model.train()
        discriminator.train()
        epoch_loss_gen = 0.0
        epoch_loss_d = 0.0

        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)
            result = diffusion_model(images, masks)

            if isinstance(result, tuple):
                loss, gen_image = result
            else:
                loss = result
                gen_image = images  # fallback for visualization
                
            d_out_fake = discriminator(gen_image)
            gan_loss_gen = criterion_gan(d_out_fake, torch.ones_like(d_out_fake))

            
            loss_total = loss + gan_loss_gen
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_gen += loss_total.item()

            # ---------------------------
            # 判别器训练
            # ---------------------------
            optimizer_d.zero_grad()
            d_out_real = discriminator(masks)
            d_loss_real = criterion_gan(d_out_real, torch.ones_like(d_out_real))
            d_out_fake = discriminator(gen_image.detach())
            d_loss_fake = criterion_gan(d_out_fake, torch.zeros_like(d_out_fake))
            loss_d = d_loss_real + d_loss_fake
            loss_d.backward()
            optimizer_d.step()
            epoch_loss_d += loss_d.item()
             
  
        avg_gen_loss = epoch_loss_gen / len(train_loader)
        avg_d_loss = epoch_loss_d / len(train_loader)

        print(f"[Diffusion] Epoch {epoch+1}/100, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_d_loss:.4f}")

        if epoch == 0 and not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'generator_loss', 'discriminator_loss'])

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_gen_loss, avg_d_loss])

        save_dir = os.path.join(weights_dir, 'diffusion')
        os.makedirs(save_dir, exist_ok=True)
        torch.save(diffusion_model.state_dict(), os.path.join(save_dir, f"diffusion_epoch_{epoch+1}.pth"))
        
        if gen_image is not None:
            gen_image = gen_image.detach().clamp(0, 1)
            
            save_dir = os.path.join(weights_dir, 'samples')
            os.makedirs(save_dir, exist_ok=True)
           
            save_path = os.path.join(save_dir, f"comparison_epoch_{epoch+1}.png") 
            save_diffusion_comparison(images, gen_image, masks, save_path, nrow=4)


        plot_losses_2(csv_file, os.path.join(weights_dir, 'loss_curve'))


            
def train_unet(args, device):
    
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    weights_dir = os.path.join(args.weights_dir, formatted_time)
    os.makedirs(weights_dir, exist_ok=True)

    csv_file = os.path.join(weights_dir, "training_unet_log.csv")

    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = UltrasoundDataset(
        image_dir=os.path.join(args.data_path, 'train', 'images'),
        mask_dir=os.path.join(args.data_path, 'train', 'masks'),
        transform=transform)

    test_dataset = UltrasoundDataset(
        image_dir=os.path.join(args.data_path, 'test', 'images'),
        mask_dir=os.path.join(args.data_path, 'test', 'masks'),
        transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNets(n_channels=1, n_classes=1).to(device)
    if hasattr(args, 'unet_ckpt') and args.unet_ckpt and os.path.exists(args.unet_ckpt):
        print("[INFO] Loading pretrained U-Net weights...")
        model.load_state_dict(torch.load(args.unet_ckpt, map_location=device))

    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    save_dir = os.path.join(weights_dir, 'unet_only')
    os.makedirs(save_dir, exist_ok=True)

    # 初始化CSV文件记录头
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader):
            images, masks = images.to(device), masks.to(device)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # 测试阶段
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)

                preds = model(images)
                loss_test = criterion(preds, masks)
                test_loss += loss_test.item()

        avg_test_loss = test_loss / len(test_loader)

        print(f"[U-Net] Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        # 每个epoch保存模型
        torch.save(model.state_dict(), os.path.join(save_dir, f"unet_epoch_{epoch+1}.pth"))

        # 每个epoch记录损失到CSV文件中
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_test_loss])

        # 绘制损失曲线
        plot_losses(csv_file, os.path.join(weights_dir, 'loss_curve.png'))
           
            
            
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.train_mode == 'diffusion_only':
        train_diffusion(args, device)
    elif args.train_mode == 'unet_only':
        train_unet(args, device)
    elif args.train_mode == 'finetune_unet':
        train_unet(args, device)
    elif args.train_mode == 'full_pipeline':
        train_diffusion(args, device)
        train_unet(args, device)


if __name__ == "__main__":
    main()

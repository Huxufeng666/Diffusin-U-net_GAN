import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data import MedicalDataset, BUSIDataset
import torch.optim as optim
import torch.nn as nn
from Discrimintor import Discriminator,ComplexDiscriminator_pro,PatchDiscriminator
from model import EndToEndModel, ResNet, BasicBlock
import os
import csv
import datetime
from tqdm import tqdm
from configs.loss import CombinedLoss,  BCEDiceLoss, DiceLoss_T, DiceLoss_v,dice_coefficient, plot_losses, plot_losses_2,plot_losses_pros
import torch.nn.functional as F
from denoising_diffusion import GaussianDiffusion, ContinuousTimeGaussianDiffusion
from U_net import UNetb, UNets,AttentionResUNet
from torchvision.utils import save_image
from configs.save_diffusion_comparison import save_diffusion_comparison, save_diffusion_comparison_2,visualize_prediction
from configs.gif import generate_grid_gif ,generate_individual_gifss
import torchvision.transforms.functional as F
import numpy as np
import glob
from configs.utils import  EarlyStopper, find_latest_diffusion_ckpt,get_save_paths
import torch.nn.functional as F




def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion + U-Net Model")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--data_path', type=str, default='/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/BUSI')
    parser.add_argument('--weights_dir', type=str, default='weights')
    parser.add_argument('--train_mode', type=str, default='full_pipeline',
                        choices=['diffusion_only', 'finetune_unet', 'unet_only', 'full_pipeline'])
    parser.add_argument('--unet_ckpt', type=str, default="", help='Path to U-Net pretrained weights')
    parser.add_argument('--diffusion_ckpt', type=str, default='', help='Path to diffusion model checkpoint')
    return parser.parse_args()




def train_diffusion(args, device,sub_module=None):
    weights_dir, csv_file = get_save_paths(args, sub_module=sub_module)   #weights/2025-04-16_20-51-20_FullPipeline

    
    # os.makedirs(weights_dir, exist_ok=True)
    csv_file = os.path.join(weights_dir, "training_diffusio_log.csv")

    # 数据增强
    image_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])

    train_dataset = BUSIDataset(
        image_dir=os.path.join(args.data_path, 'train', 'images'),
        mask_dir=os.path.join(args.data_path, 'train', 'masks'),
        image_transform=image_transform,
        mask_transform=mask_transform)

    val_dataset = BUSIDataset(
        image_dir=os.path.join(args.data_path, 'val', 'images'),
        mask_dir=os.path.join(args.data_path, 'val', 'masks'),
        image_transform=image_transform,
        mask_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    sample_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    sample_images, sample_masks = next(iter(sample_loader))
    sample_images, sample_masks = sample_images.to(device), sample_masks.to(device)

    diffusion_model = ContinuousTimeGaussianDiffusion(
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
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    early_stopper = EarlyStopper(patience=20, min_delta=1e-4, model=diffusion_model, path=os.path.join(weights_dir, "best_model"))



    sample_dir = os.path.join(weights_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)


    for epoch in range(args.epochs):
        diffusion_model.train()
        discriminator.train()
        epoch_loss_gen = 0.0
        epoch_loss_d = 0.0


        for step, (images, masks) in enumerate(tqdm(train_loader)):
            images, masks = images.to(device), masks.to(device)
            result = diffusion_model(images, mask=masks)

            if isinstance(result, tuple):
                loss, gen_image = result
            else:
                loss = result
                gen_image = images
                
            

            d_out_fake = discriminator(images)
            gan_loss_gen = criterion_gan(d_out_fake, torch.ones_like(d_out_fake))
            loss_total = loss + gan_loss_gen

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            epoch_loss_gen += loss_total.item()

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

        # ========= 验证集评估 =========
        diffusion_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                result = diffusion_model(images, masks)
                loss = result[0] if isinstance(result, tuple) else result
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"[Epoch {epoch+1}] Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_d_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # ========= 保存 CSV =========
        with open(csv_file, 'a' if epoch else 'w', newline='') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(['epoch', 'generator_loss', 'discriminator_loss', 'val_loss'])
            writer.writerow([epoch + 1, avg_gen_loss, avg_d_loss, avg_val_loss])


        # # ✅ 保存最佳模型
        if early_stopper.best_loss is None or avg_val_loss < early_stopper.best_loss - early_stopper.min_delta:

            with open(os.path.join(weights_dir, "best_epoch.txt"), 'w') as f:
                f.write(str(epoch + 1))

        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                gen_vis = diffusion_model(sample_images, sample_masks)[1].clamp(0, 1)
                for i in range(4):
                    save_path = os.path.join(sample_dir, f"pred_epoch{epoch+1}_sample{i+1}.png")
                    visualize_prediction(sample_images[i].cpu(), sample_masks[i].cpu(), gen_vis[i].cpu(), save_path=save_path)

        gif_output_path = os.path.join(weights_dir, "prediction_progress")
        os.makedirs(gif_output_path, exist_ok=True)
        generate_individual_gifss(image_folder=sample_dir, output_folder=gif_output_path, duration=5)

        plot_losses_2(csv_file, os.path.join(weights_dir, 'loss_curve'))

        # ========= 提前停止 =========
        if early_stopper(avg_val_loss,epoch=epoch + 1):
            print(f"[⛔] Early stopping triggered. Best val loss: {early_stopper.best_loss:.4f}")
            break



def train_unet(args, device, use_diffusion_input=False, diffusion_model=None,sub_module=None):
 
    
    weights_dir, csv_file = get_save_paths(args, sub_module=sub_module)  
 
    # weights_dir = os.path.join(weights_unet, "_U-Net")
    # os.makedirs(weights_dir, exist_ok=True)
    csv_file = os.path.join(weights_dir, "training_U-Net_log.csv")


    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = BUSIDataset(
        image_dir=os.path.join(args.data_path, 'train', 'images'),
        mask_dir=os.path.join(args.data_path, 'train', 'masks'),
        image_transform=transform,
        mask_transform=mask_transform
    )
    val_dataset = BUSIDataset(
        image_dir=os.path.join(args.data_path, 'val', 'images'),
        mask_dir=os.path.join(args.data_path, 'val', 'masks'),
        image_transform=transform,
        mask_transform=mask_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model =AttentionResUNet().to(device)

    save_dir = os.path.join(weights_dir, 'unet_only')
    os.makedirs(save_dir, exist_ok=True)  # 这行必须在 EarlyStopper 之前

    early_stopper = EarlyStopper(
        patience=20,
        min_delta=1e-4,
        model=model,
        path=os.path.join(save_dir, "best_model.pth")
    )

    if args.unet_ckpt and os.path.exists(args.unet_ckpt):
        model.load_state_dict(torch.load(args.unet_ckpt, map_location=device))
        print("[INFO] Loaded U-Net from:", args.unet_ckpt)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion_t = DiceLoss_T()
    criterion_v = CombinedLoss()

    sample_dir = os.path.join(weights_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)
    gif_output_path = os.path.join(weights_dir, "prediction_progress.gif")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Dice'])

    fixed_images_batch, fixed_masks_batch = next(iter(val_loader))
    fixed_images = fixed_images_batch[:4].to(device)
    fixed_masks = fixed_masks_batch[:4].to(device)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, (images, masks) in enumerate(tqdm(train_loader)):
            masks = masks.to(device)
            if use_diffusion_input:
                with torch.no_grad():
                    _, images = diffusion_model(images.to(device), masks)
            else:
                images = images.to(device)
            preds = model(images)
            loss = criterion_t(preds, masks.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 可视化输出
        model.eval()
        with torch.no_grad():
            fixed_preds = torch.sigmoid(model(fixed_images))
            for i in range(4):
                save_path = os.path.join(sample_dir, f"pred_epoch{epoch+1}_sample{i+1}.png")
                pred_i = fixed_preds[i].squeeze(0)
                visualize_prediction(fixed_images[i].cpu(), fixed_masks[i].cpu(), pred_i, save_path=save_path)

        generate_individual_gifss(image_folder=sample_dir, output_folder=gif_output_path)

        avg_train_loss = total_loss / len(train_loader)

        val_loss = 0.0
        dice_scores = []
        with torch.no_grad():
            for images, masks in val_loader:
                masks = masks.to(device)
                if use_diffusion_input:
                    _, images = diffusion_model(images.to(device), masks)
                else:
                    images = images.to(device)
                preds = torch.sigmoid(model(images))
                loss = criterion_v(preds, masks)
                val_loss += loss.item()
                dice = dice_coefficient((preds > 0.5).float().cpu().numpy(), masks.cpu().numpy())
                dice_scores.append(dice)

        avg_val_loss = val_loss / len(val_loader)
        avg_dice = np.mean(dice_scores)

        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Dice: {avg_dice:.4f}")

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_dice])

        plot_losses(csv_file, os.path.join(weights_dir, 'loss_curve_total.png'))
        plot_losses_pros(csv_file, os.path.join(weights_dir, 'loss_curve.png'))

        if early_stopper(avg_val_loss,epoch=epoch+1):
            print("[⛔] Early stopping triggered. Best val loss:", early_stopper.best_loss)
            break




def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args = parse_args()
    args.timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  

    if args.train_mode == 'diffusion_only':
        train_diffusion(args, device, sub_module='diffusion')

    elif args.train_mode == 'unet_only':
        train_unet(args, device, use_diffusion_input=False, diffusion_model=None, sub_module='unet')

    elif args.train_mode == 'finetune_unet':
        diffusion_model = GaussianDiffusion(
            model=ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1),
            image_size=256,
            timesteps=1000,
            objective='pred_noise',
            beta_schedule='sigmoid'
        ).to(device)
        diffusion_model.load_state_dict(torch.load(args.diffusion_ckpt, map_location=device))
        train_unet(args, device, use_diffusion_input=True, diffusion_model=diffusion_model, sub_module='unet')

    elif args.train_mode == 'full_pipeline':
        train_diffusion(args, device, sub_module='diffusion')
        best_ckpt = find_latest_diffusion_ckpt(args.weights_dir)
        diffusion_model = GaussianDiffusion(
            model=ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1),
            image_size=256,
            timesteps=1000,
            objective='pred_noise',
            beta_schedule='sigmoid'
        ).to(device)
        if best_ckpt:
            diffusion_model.load_state_dict(torch.load(best_ckpt, map_location=device))
            print(f"[INFO] Loaded best diffusion model from: {best_ckpt}")
        else:
            print("[WARN] No best diffusion model found, using random initialization.")
        train_unet(args, device, use_diffusion_input=True, diffusion_model=diffusion_model, sub_module='unet')


if __name__ == '__main__':
    main()

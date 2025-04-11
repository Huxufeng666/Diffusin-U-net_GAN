import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data import MedicalDataset, BUSDataset,BUS_UCMDataset,BUSIDataset
import torch.optim as optim
import torch.nn as nn
from Discrimintor import Discriminator,ComplexDiscriminator_pro
from model import EndToEndModel, ResNet, BasicBlock
import os
import csv
import datetime
from tqdm import tqdm
from configs.loss import CombinedLoss, CombinedLoss_pro, BCEDiceLoss, DiceLoss_T, DiceLoss_v,dice_coefficient, plot_losses, plot_losses_2,plot_losses_pros
import torch.nn.functional as F
from denoising_diffusion import GaussianDiffusion
from UNet import UNet, UNets,AttentionResUNet
# from torchvision.utils import save_image
# from torchvision.utils import make_grid, save_image
import numpy as np
from configs.save_diffusion_comparison import save_diffusion_comparison, save_diffusion_comparison_2,visualize_prediction
from configs.gif import generate_grid_gif ,generate_individual_gifss



def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion + U-Net Model")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--data_path', type=str, default='/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/BUSI')
    parser.add_argument('--weights_dir', type=str, default='weights')
    parser.add_argument('--train_mode', type=str, default='diffusion_only', choices=['diffusion_only', 'finetune_unet', 'unet_only', 'full_pipeline'])
    parser.add_argument('--unet_ckpt', type=str, default="", help='Path to U-Net pretrained weights')
    parser.add_argument('--diffusion_ckpt', type=str, default='', help='Path to diffusion model checkpoint')
    return parser.parse_args()


class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            return False

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop

# ========== Train Function ==========

def train_diffusion(args, device):
    import datetime, os, csv
    from tqdm import tqdm
    from glob import glob
    from PIL import Image
    import numpy as np
    import imageio

    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    weights_dir = os.path.join(args.weights_dir, formatted_time + "_Diffusion")
    os.makedirs(weights_dir, exist_ok=True)
    csv_file = os.path.join(weights_dir, "training_diffusio_log.csv")

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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    sample_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    sample_iter = iter(sample_loader)
    sample_images, sample_masks = next(sample_iter)
    sample_images, sample_masks = sample_images.to(device), sample_masks.to(device)

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

    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=args.lr)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    early_stopper = EarlyStopper(patience=10, min_delta=1e-4)
    print("[INFO] Starting diffusion training...")

    sample_dir = os.path.join(weights_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)
    top_k = 5
    top_checkpoints = []

    for epoch in range(args.epochs):
        diffusion_model.train()
        discriminator.train()
        epoch_loss_gen = 0.0
        epoch_loss_d = 0.0

        for step, (images, masks) in enumerate(tqdm(train_loader)):
            images, masks = images.to(device), masks.to(device)
            result = diffusion_model(images, masks)

            if isinstance(result, tuple):
                loss, gen_image = result
            else:
                loss = result
                gen_image = images

            d_out_fake = discriminator(gen_image)
            gan_loss_gen = criterion_gan(d_out_fake, torch.ones_like(d_out_fake))
            loss_total = loss + gan_loss_gen

            optimizer.zero_grad()
            loss.backward()
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

        print(f"[Diffusion] Epoch {epoch+1}/{args.epochs}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_d_loss:.4f}")

        if epoch == 0 and not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'generator_loss', 'discriminator_loss'])

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_gen_loss, avg_d_loss])

        save_dir = os.path.join(weights_dir, 'diffusion')
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"diffusion_epoch_{epoch+1}.pth")
        torch.save(diffusion_model.state_dict(), model_path)

        top_checkpoints.append((avg_gen_loss, model_path))
        top_checkpoints.sort(key=lambda x: x[0])
        if len(top_checkpoints) > top_k:
            worst_loss, worst_path = top_checkpoints.pop()
            if os.path.exists(worst_path):
                os.remove(worst_path)

        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                gen_vis = diffusion_model(sample_images, sample_masks)[1]
                gen_vis = gen_vis.detach().clamp(0, 1)
                for i in range(4):
                    save_path = os.path.join(sample_dir, f"pred_epoch{epoch+1}_sample{i+1}.png")
                    visualize_prediction(
                        sample_images[i].cpu(),
                        sample_masks[i].cpu(),
                        gen_vis[i].cpu(),
                        save_path=save_path
                    )
        gif_output_path = os.path.join(weights_dir, "prediction_progress")
        os.makedirs(gif_output_path, exist_ok=True)
        generate_individual_gifss(image_folder=sample_dir, output_folder=gif_output_path, duration=5)

        plot_losses_2(csv_file, os.path.join(weights_dir, 'loss_curve'))

        # if early_stopper(avg_gen_loss):
        #     print("[⛔] Early stopping triggered. Best loss:", early_stopper.best_loss)
        #     break


        
def train_unet(args, device):
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    weights_dir = os.path.join(args.weights_dir, formatted_time +"_U-net")
    os.makedirs(weights_dir, exist_ok=True)

    csv_file = os.path.join(weights_dir, "training_unet_log.csv")

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

    test_dataset = BUSIDataset(
        image_dir=os.path.join(args.data_path, 'test', 'images'),
        mask_dir=os.path.join(args.data_path, 'test', 'masks'),
        image_transform=transform,
        mask_transform=mask_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNets(n_channels=1,n_classes=1).to(device)
    # model = AttentionResUNet().to(device)
    if hasattr(args, 'unet_ckpt') and args.unet_ckpt and os.path.exists(args.unet_ckpt):
        print("[INFO] Loading pretrained U-Net weights...")
        model.load_state_dict(torch.load(args.unet_ckpt, map_location=device))

    criterion_t = DiceLoss_T()
    criterion_v = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    save_dir = os.path.join(weights_dir, 'unet_only')
    os.makedirs(save_dir, exist_ok=True)

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Dice'])

    early_stopper = EarlyStopper(patience=10, min_delta=1e-4)
    top_k = 5
    top_checkpoints = []

    fixed_images_batch, fixed_masks_batch = next(iter(test_loader))
    fixed_images = fixed_images_batch[:4].to(device)
    fixed_masks = fixed_masks_batch[:4].to(device)

    sample_dir = os.path.join(weights_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for step, (images, masks) in enumerate(tqdm(train_loader)):
            images, masks = images.to(device), masks.to(device)
            masks = masks.float()
            preds = model(images)
            loss = criterion_t(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if epoch % 1 == 0 and step == 0:
                model.eval()
                with torch.no_grad():
                    fixed_preds = model(fixed_images) 
                    fixed_preds = torch.sigmoid(fixed_preds)  # 如果模型输出是 logits
                             
                    for i in range(4):
                        # visualize_prediction(fixed_images[i], fixed_masks[i], fixed_preds[i], save_path=save_path)
                        
                        save_path = os.path.join(sample_dir, f"pred_epoch{epoch+1}_sample{i+1}.png")
                        pred_i = fixed_preds[i].squeeze(0)  # [H, W]
                        # pred_i = (pred_i > 0.5).float() # 二值化
                        
                        visualize_prediction(
                                fixed_images[i].cpu(),
                                fixed_masks[i].cpu(),
                                pred_i,
                                save_path=save_path
                            )
                                                            
            
        gif_output_path = os.path.join(weights_dir, "prediction_progress.gif")

        generate_individual_gifss(image_folder=os.path.join(weights_dir, "sample"), output_folder=gif_output_path)

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0.0
        dice_scores = []

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                preds = torch.sigmoid(model(images))
                loss_test = criterion_v(preds, masks)
                test_loss += loss_test.item()
                preds_bin = (preds > 0.5).float()
                dice = dice_coefficient(preds_bin.detach().cpu().numpy(), masks.detach().cpu().numpy())
                dice_scores.append(dice)

        avg_test_loss = test_loss / len(test_loader)
        avg_dice = np.mean(dice_scores)

        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Dice: {avg_dice:.4f}")

        model_path = os.path.join(save_dir, f"unet_epoch_{epoch+1}_dice_{avg_dice:.2f}.pth")
        torch.save(model.state_dict(), model_path)

        # top_checkpoints.append((avg_test_loss, model_path))
        # top_checkpoints.sort(key=lambda x: x[0])
        # if len(top_checkpoints) > top_k:
        #     worst_loss, worst_path = top_checkpoints.pop()
        #     if os.path.exists(worst_path):
        #         os.remove(worst_path)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_test_loss, avg_dice])

        plot_losses(csv_file, os.path.join(weights_dir, 'loss_curve_total.png'))
        plot_losses_pros(csv_file, os.path.join(weights_dir, 'loss_curve.png'))

        if early_stopper(avg_test_loss):
            print("[⛔] Early stopping triggered. Best loss:", early_stopper.best_loss)
            break


            
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

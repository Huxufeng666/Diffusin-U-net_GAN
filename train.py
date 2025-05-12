import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data import MedicalDataset,UltrasoundDataset
import torch.optim as optim
import torch.nn as nn
from  Discrimintor import Discriminator
from model import EndToEndModel
import csv
import os
import datetime
from tqdm import tqdm
from loss import CombinedLoss,compute_edge_from_mask, plot_losses,plot_losses_2
import torch.nn.functional as F
from monai.losses import HausdorffDTLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion + U-Net Model")

    # 添加命令行参数
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate for the optimizer')
    parser.add_argument('--size', type=float, default=256, help='input images size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to train the model on')
    parser.add_argument('--data_path', type=str, default='/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/BUS-UCLM/partitions', help='Base directory of the dataset')
    parser.add_argument('--weights_dir', type=str, default='weights', help='Directory to save model weights')

    return parser.parse_args()





def main():
    # 解析命令行参数
    args = parse_args()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((args.size,args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    mask_transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor()
])

    # 数据集路径
    # train_image_dir = os.path.join(args.data_path, 'train', 'original')
    # train_mask_dir  = os.path.join(args.data_path, 'train', 'mask')
    # test_image_dir  = os.path.join(args.data_path, 'val', 'original')
    # test_mask_dir   = os.path.join(args.data_path, 'val', 'mask')

    
    train_image_dir = os.path.join(args.data_path, 'train', 'images')
    train_mask_dir  = os.path.join(args.data_path, 'train', 'masks')
    test_image_dir  = os.path.join(args.data_path, 'test', 'images')
    test_mask_dir   = os.path.join(args.data_path, 'test', 'masks')
    
    # # # 创建数据集实例
    # train_dataset = MedicalDataset(image_dir=train_image_dir, mask_dir=train_mask_dir, transform=transform)
    # test_dataset  = MedicalDataset(image_dir=test_image_dir, mask_dir=test_mask_dir, transform=transform)

    train_dataset = UltrasoundDataset(image_dir=train_image_dir, mask_dir=train_mask_dir, transform=transform)#,mask_transform=mask_transform)
    test_dataset  = UltrasoundDataset(image_dir=test_image_dir, mask_dir=test_mask_dir, transform=transform)#,mask_transform=mask_transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化设备、模型和优化器
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 模型和判别器
    model = EndToEndModel().to(device)
    discriminator = Discriminator().to(device)


    criterion_seg =nn.MSELoss()   # 分割损失
    criterion_gan = CombinedLoss(alpha=1.0, beta=1.0, gamma=0.2)  # GAN 对抗损失
    # criterion_mse = nn.MSELoss()            # 可选的额外约束
    criterion_edge =nn.BCEWithLogitsLoss()  # 边缘损失
    criterion_hausdorff = HausdorffDTLoss()  # 形态损失
    
    # 定义优化器
    optimizer_model = optim.NAdam(model.parameters(), lr=args.lr)
    optimizer_d = optim.AdamW(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # ------------------------------
    # 初始化权重保存路径
    # ------------------------------
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    weights_dir = os.path.join(args.weights_dir, formatted_time)
    os.makedirs(weights_dir, exist_ok=True)

    # CSV 文件保存路径及初始化写入表头
    csv_file = os.path.join(weights_dir, "training_log.csv")
    csv_header = ['epoch', 'generator_loss', 'discriminator_loss','Test Seg Loss']
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    # ------------------------------
    # 训练循环
    # ------------------------------
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        model.train()
        discriminator.train()
        epoch_loss_gen = 0.0
        epoch_loss_d = 0.0

        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            gt_mask = masks.to(device)

            # ---------------------------
            # 生成器前向传播
            # ---------------------------
            processed_image, segmentation = model(images, masks)
            
            segmentation= torch.sigmoid(segmentation)
            gt_mask = compute_edge_from_mask(masks)
            loss_seg = criterion_seg(segmentation, gt_mask).to(device)
            
            # loss_seg = criterion_edge(edge_pred, gt_edge)
            
            d_out_fake = discriminator(segmentation)
            
            
            gan_loss_gen = criterion_gan(d_out_fake, torch.ones_like(d_out_fake))
            loss_gen = loss_seg + gan_loss_gen

            loss_hd = criterion_hausdorff(segmentation, masks)
            loss_total = loss_gen + loss_hd 
            
            
            optimizer_model.zero_grad()
            loss_gen.backward()
            optimizer_model.step()
            epoch_loss_gen += loss_total.item()

            # ---------------------------
            # 判别器训练
            # ---------------------------
            optimizer_d.zero_grad()
            d_out_real = discriminator(gt_mask)
            d_loss_real = criterion_gan(d_out_real, torch.ones_like(d_out_real))
            d_out_fake = discriminator(segmentation.detach())
            d_loss_fake = criterion_gan(d_out_fake, torch.zeros_like(d_out_fake))
            loss_d = d_loss_real + d_loss_fake

            loss_d.backward()
            optimizer_d.step()
            epoch_loss_d += loss_d.item()

        avg_gen_loss = epoch_loss_gen / len(train_loader)
        avg_d_loss = epoch_loss_d / len(train_loader)

        # 评估
        if (epoch + 1) % 1 == 0:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for images, masks in test_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    gt_mask = masks
                    _, segmentation = model(images, masks)
                    loss_test = criterion_seg(segmentation, gt_mask)
                    test_loss += loss_test.item()
            avg_test_loss = test_loss / len(test_loader)
        else:
            avg_test_loss = "NA"

        print(f"Epoch [{epoch+1}/{num_epochs}] Generator Loss: {avg_gen_loss:.4f}, "
              f"Discriminator Loss: {avg_d_loss:.4f}, Test Seg Loss: {avg_test_loss}")

        torch.save(model.state_dict(), os.path.join(weights_dir, f"model_epoch_{epoch+1}.pth"))
        
        plot_losses_2(csv_file, os.path.join( weights_dir,'loss_curve.png'))
        
        # 保存训练日志
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_gen_loss, avg_d_loss,avg_test_loss])

if __name__ == "__main__":
    main()


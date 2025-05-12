

import argparse
import os
import datetime
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import UNets  
from U_net import  AttentionResUNet# 确保你已经导入了 UNet 模型类


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained U-Net model")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--unet_ckpt', type=str, default='weights/2025-05-12_01-59-05_UNet/unet_only/best_model.pth/epoch_36_best_model.pth', help='Path to trained U-Net weights')
    parser.add_argument('--image_path', type=str, default="/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/BUS-UCLM/test/images/UNCU_003.png",help='Path to test image or directory of images')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save predictions')
    parser.add_argument('--size', type=int, default=256)
    return parser.parse_args()


def load_and_predict(model, image_path, transform, device, output_dir, size):
    image = Image.open(image_path).convert("L")
    original_size = image.size
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        output_resized = F.interpolate(output, size=original_size[::-1], mode='bilinear', align_corners=False)
        pred = (output_resized > 0.5).float().cpu().squeeze().numpy()

    # 可视化与保存
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(output_dir, f"{filename}_pred_{now}.png")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Segmentation")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"[✔] Saved: {save_path}")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = AttentionResUNet().to(device)
    model.load_state_dict(torch.load(args.unet_ckpt, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 判断是单图还是文件夹
    if os.path.isdir(args.image_path):
        image_list = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"[INFO] Found {len(image_list)} images in folder: {args.image_path}")
    else:
        image_list = [args.image_path]

    for image_path in image_list:
        load_and_predict(model, image_path, transform, device, args.output_dir, args.size)


if __name__ == "__main__":
    main()


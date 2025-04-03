import argparse
import torch
from torchvision import transforms
from PIL import Image
import os
import datetime
import matplotlib.pyplot as plt
from UNet import UNets
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Test trained U-Net model")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--unet_ckpt', type=str,  default='weights/2025-04-03_13-57-13/unet_only/unet_epoch_200.pth', help='Path to trained U-Net weights')
    parser.add_argument('--image_path', type=str, default='/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/BUS-UCLM/partitions/test/images/ANFO_005.png', help='Path to test image')
    parser.add_argument('--output_dir', type=str, default='weights/2025-04-03_13-57-13', help='Directory to save predictions')
    parser.add_argument('--size', type=int, default=256)
    return parser.parse_args()


# 加载模型权重并进行预测
def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = UNets(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(args.unet_ckpt, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 创建输出文件夹
    os.makedirs(args.output_dir, exist_ok=True)

    image = Image.open(args.image_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()

    pred = output.cpu().squeeze().numpy()

    # 保存预测结果
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(args.image_path).split('.')[0]
    output_path = os.path.join(args.output_dir, f"{filename}_pred_{now}.png")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Segmentation")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.savefig(output_path)
    print(f"[INFO] Prediction saved to {output_path}")


if __name__ == "__main__":
    main()

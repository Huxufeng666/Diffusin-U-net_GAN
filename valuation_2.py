import argparse
import torch
import numpy as np
import csv
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from data import UltrasoundDataset,MedicalDataset
from UNet import UNets
from tqdm import tqdm

# 指标计算函数
def threshold_predictions(predictions, threshold=0.5):
    return (predictions >= threshold).astype(np.float32)

def dice_coefficient(pred, target, smooth=1e-6):
    intersection = np.sum(pred * target)
    return (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

def iou_score(pred, target, smooth=1e-6):
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(pred, target):
    correct = np.sum(pred == target)
    return correct / pred.size

def precision_score(pred, target, smooth=1e-6):
    TP = np.sum(pred * target)
    FP = np.sum(pred) - TP
    return (TP + smooth) / (TP + FP + smooth)

def recall_score(pred, target, smooth=1e-6):
    TP = np.sum(pred * target)
    FN = np.sum(target) - TP
    return (TP + smooth) / (TP + FN + smooth)

# 测试函数
def main():
    parser = argparse.ArgumentParser(description="Evaluate U-Net Model")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--unet_ckpt', type=str,  default="weights/2025-04-04_11-25-35_U-net/unet_only/unet_epoch_200_dice_0.2000.pth", help='Path to trained U-Net weights')
    parser.add_argument('--data_path', type=str, default="/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/BUSI", help='Path to dataset root directory')
    parser.add_argument('--output_dir', type=str, default='weights/2025-04-04_11-25-35_U-net', help='Directory to save evaluation results')
    parser.add_argument('--size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = UNets(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(args.unet_ckpt, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_dataset = MedicalDataset(
        image_dir=os.path.join(args.data_path, 'test', 'images'),
        mask_dir=os.path.join(args.data_path, 'test', 'masks'),
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    dice_list, iou_list, pix_acc_list, precision_list, recall_list = [], [], [], [], []

    with torch.no_grad():
        for images, masks in tqdm(test_loader):
            images, masks = images.to(device), masks.to(device)
            preds = torch.sigmoid(model(images))

            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for pred, mask in zip(preds_np, masks_np):
                pred_bin = threshold_predictions(pred.squeeze())
                mask_bin = threshold_predictions(mask.squeeze())

                dice_list.append(dice_coefficient(pred_bin, mask_bin))
                iou_list.append(iou_score(pred_bin, mask_bin))
                pix_acc_list.append(pixel_accuracy(pred_bin, mask_bin))
                precision_list.append(precision_score(pred_bin, mask_bin))
                recall_list.append(recall_score(pred_bin, mask_bin))

    os.makedirs(args.output_dir, exist_ok=True)

    avg_metrics = {
        "Dice Coefficient": np.mean(dice_list),
        "IoU": np.mean(iou_list),
        "Pixel Accuracy": np.mean(pix_acc_list),
        "Precision": np.mean(precision_list),
        "Recall": np.mean(recall_list)
    }

    csv_path = os.path.join(args.output_dir, "evaluation_metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for metric, value in avg_metrics.items():
            writer.writerow([metric, value])

    print(f"Evaluation results saved to {csv_path}")

if __name__ == "__main__":
    main()

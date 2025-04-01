import torch
import torch.nn as nn
import numpy as np
import csv
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from  data  import  MedicalDataset 
from model import EndToEndModel
from tqdm import tqdm

# 以下函数均假设输入为 numpy 数组，且 shape 为 (H, W)
def threshold_predictions(predictions, threshold=0.5):
    """
    将预测概率图转换为二值图
    """
    return (predictions >= threshold).astype(np.float32)

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Dice 系数计算公式：
        Dice = (2*TP + smooth) / (2*TP + FP + FN + smooth)
    """
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = np.sum(pred * target)
    return (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

def iou_score(pred, target, smooth=1e-6):
    """
    IoU（交并比）计算公式：
        IoU = (TP + smooth) / (TP + FP + FN + smooth)
    """
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    return (intersection + smooth) / (union + smooth)

def pixel_accuracy(pred, target):
    """
    像素准确率计算公式：
        Pixel Accuracy = (TP+TN) / Total Pixels
    """
    correct = np.sum(pred == target)
    total = pred.size
    return correct / total

def precision_score(pred, target, smooth=1e-6):
    """
    精确率：TP / (TP + FP)
    """
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    TP = np.sum(pred * target)
    FP = np.sum(pred) - TP
    return (TP + smooth) / (TP + FP + smooth)

def recall_score(pred, target, smooth=1e-6):
    """
    召回率：TP / (TP + FN)
    """
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    TP = np.sum(pred * target)
    FN = np.sum(target) - TP
    return (TP + smooth) / (TP + FN + smooth)

# 假设 MedicalDataset、EndToEndModel 和 Discriminator 类已经定义好
# 定义预处理操作（与训练时一致）
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 数据集路径，请根据实际情况修改
test_image_dir  = '/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/dataset/test/original'
test_mask_dir   = '/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/dataset/test/mask'

# 创建测试集数据集和 DataLoader
test_dataset  = MedicalDataset(image_dir=test_image_dir, mask_dir=test_mask_dir, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 初始化设备、模型并加载训练好的权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = EndToEndModel().to(device)
# 请根据实际情况指定权重文件路径
checkpoint_path = "weights/2025-03-25 08:52:04/model_epoch_200.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# 统计所有测试数据的指标
dice_list = []
iou_list = []
pix_acc_list = []
precision_list = []
recall_list = []

with torch.no_grad():
    for images, masks in tqdm(test_loader):
        images = images.to(device)
        masks = masks.to(device)
        # 模型推理：得到分割输出，注意 U-Net 输出未经过 Sigmoid
        _, segmentation = model(images, masks)
        segmentation = torch.sigmoid(segmentation)
        
        # 将 Tensor 转换为 numpy 数组
        seg_np_batch = segmentation.cpu().numpy()  # shape: (B, 1, H, W)
        gt_np_batch = masks.cpu().numpy()            # shape: (B, 1, H, W)
        
        batch_size = seg_np_batch.shape[0]
        for i in range(batch_size):
            # squeeze 去除 channel 维度
            pred = seg_np_batch[i].squeeze()  # (H, W) 浮点数 [0,1]
            target = gt_np_batch[i].squeeze() # (H, W) 浮点数，如果 mask 原本是归一化到 [0,1] 或 0/1
            
            # 二值化预测（阈值0.5）
            pred_bin = threshold_predictions(pred, threshold=0.5)
            target_bin = threshold_predictions(target, threshold=0.5)
            
            dice_val = dice_coefficient(pred_bin, target_bin)
            iou_val = iou_score(pred_bin, target_bin)
            pix_acc = pixel_accuracy(pred_bin, target_bin)
            prec = precision_score(pred_bin, target_bin)
            rec = recall_score(pred_bin, target_bin)
            
            dice_list.append(dice_val)
            iou_list.append(iou_val)
            pix_acc_list.append(pix_acc)
            precision_list.append(prec)
            recall_list.append(rec)

# 计算平均指标
avg_dice = np.mean(dice_list)
avg_iou = np.mean(iou_list)
avg_pix_acc = np.mean(pix_acc_list)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)

print("Average Dice Coefficient:", avg_dice)
print("Average IoU:", avg_iou)
print("Average Pixel Accuracy:", avg_pix_acc)
print("Average Precision:", avg_precision)
print("Average Recall:", avg_recall)

# ------------------------------
# 将结果保存到 CSV 文件中
# ------------------------------
checkpoint_dir = os.path.dirname(checkpoint_path)

output_csv = os.path.join(checkpoint_dir, "segmentation_evaluation.csv")

with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Dice Coefficient", avg_dice])
    writer.writerow(["IoU", avg_iou])
    writer.writerow(["Pixel Accuracy", avg_pix_acc])
    writer.writerow(["Precision", avg_precision])
    writer.writerow(["Recall", avg_recall])

print("Evaluation results saved to", output_csv)

import os
import cv2
import pandas as pd
import numpy as np

def load_dataset(csv_path, image_dir, fold=1):
    """
    从 CSV 文件中加载数据，并根据指定的验证折（fold）划分出训练集和验证集。
    
    参数：
      csv_path: CSV 文件路径
      image_dir: 图像存放目录，假设图像文件命名为 <ID>.jpg
      fold: 指定验证折（1~5），在该折中，CSV 中 valid_i==1 的样本作为验证集，
            而 valid_i 为 NaN 的样本作为训练集
    返回：
      train_paths: 训练集图像路径列表
      valid_paths: 验证集图像路径列表
      train_labels: 训练集对应的病理标签列表
      valid_labels: 验证集对应的病理标签列表
    """
    df = pd.read_csv(csv_path)
    valid_col = f'valid_{fold}'
    
    # 样本在当前折中标记为 1 的作为验证样本
    df_valid = df[df[valid_col] == 1].reset_index(drop=True)
    # 未标记为验证的作为训练样本
    df_train = df[df[valid_col].isna()].reset_index(drop=True)
    
    # 构造文件路径及标签
    train_paths = [os.path.join(image_dir, f"{row['ID']}.jpg") for _, row in df_train.iterrows()]
    valid_paths = [os.path.join(image_dir, f"{row['ID']}.jpg") for _, row in df_valid.iterrows()]
    train_labels = df_train['Pathology'].tolist()
    valid_labels = df_valid['Pathology'].tolist()
    
    return train_paths, valid_paths, train_labels, valid_labels

def load_image(image_path):
    """
    使用 OpenCV 读取图像并转换为 RGB 格式（假设原始图像为 BGR）。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法找到图像：{image_path}")
    # OpenCV 读取的图像默认是 BGR 格式，转换为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

if __name__ == '__main__':
    # CSV 文件路径与图像文件夹
    csv_path = "5-fold-cv.csv"
    image_dir = "images"  # 假设图像存放在 images 目录下
    
    # 指定当前使用的验证折（例如 fold=1）
    fold = 1
    train_paths, valid_paths, train_labels, valid_labels = load_dataset(csv_path, image_dir, fold)
    
    # 输出部分信息查看
    print("训练集样本数量：", len(train_paths))
    print("验证集样本数量：", len(valid_paths))
    
    # 加载第一张训练图像进行预览
    if train_paths:
        img_train = load_image(train_paths[0])
        print("第一张训练图像尺寸：", img_train.shape)
    
    # 加载第一张验证图像进行预览
    if valid_paths:
        img_valid = load_image(valid_paths[0])
        print("第一张验证图像尺寸：", img_valid.shape)
    
    # 如果需要，你可以进一步将数据构造成 PyTorch Dataset，
    # 然后使用 DataLoader 进行批量训练和验证

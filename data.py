# from torch.utils.data import Dataset
# from PIL import Image
# import os

# class MedicalDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         """
#         :param image_dir: 存放医学影像的目录
#         :param mask_dir: 存放掩码的目录
#         :param transform: 图像预处理操作（如调整大小、归一化等）
#         """
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.image_paths = sorted(os.listdir(image_dir))
#         self.mask_paths = sorted(os.listdir(mask_dir))
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         # 加载影像和对应掩码（均转换为灰度图）
#         image_path = os.path.join(self.image_dir, self.image_paths[idx])
#         mask_path = os.path.join(self.mask_dir, self.mask_paths[idx])
#         image = Image.open(image_path).convert("L")
#         mask = Image.open(mask_path).convert("L")

#         if self.transform:
#             image = self.transform(image)
#             mask = self.transform(mask)
#         return image, mask


from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2


class MedicalDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        :param image_dir: 存放医学影像的目录
        :param mask_dir: 存放掩码的目录
        :param transform: 图像预处理操作（如调整大小、归一化等）
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载影像
        image_name = self.image_paths[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("L")
        
        # 根据图像文件名确定对应的 mask 文件名
        base_name, ext = os.path.splitext(image_name)
        mask_candidate1 = os.path.join(self.mask_dir, f"{base_name}_mask_1.png")
        mask_candidate2 = os.path.join(self.mask_dir, f"{base_name}_mask.png")
        
        if os.path.exists(mask_candidate1):
            mask_path = mask_candidate1
        elif os.path.exists(mask_candidate2):
            mask_path = mask_candidate2
        else:
            raise FileNotFoundError(f"没有找到图像 {image_name} 对应的 mask 文件！")
        
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

class BUSIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("L")

        base_name, _ = os.path.splitext(image_name)
        mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.png")  # 或按实际命名匹配

        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask




class BUSBRADataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.image_files = sorted(os.listdir(image_dir))  # 图像文件名列表

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('bus', 'mask'))

        image = Image.open(img_path).convert('L')   # 灰度图像
        mask = Image.open(mask_path).convert('L')    # 二值图像

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
    

class BUS_UCMDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 获取图像路径
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)  # 假设 mask 文件名与 image 相同

        # 打开图像和 mask（都为灰度图）
        image = Image.open(image_path).convert("L")
        
        # base_name, ext = os.path.splitext(image_name)
        
        # mask_path = os.path.join(self.mask_dir, f"{base_name}_mask.png")
        mask = Image.open(mask_path).convert("L")

        
        # 图像预处理
        # if self.transform:
        #     image = self.transform(image)
        # if self.mask_transform:
        #     mask = self.mask_transform(mask)
        if self.image_transform:
            image = self.image_transform(image)
            mask = self.mask_transform(mask)

        return image, mask
    


class BUS_UCLM_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.transform = transform

        # 使用单独的 mask 处理方式
        self.mask_transform = mask_transform if mask_transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())  # 二值化
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

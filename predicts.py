
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import EndToEndModel
import os
import torch.nn.functional as F
import cv2
import datetime
# 假设你的端到端模型类定义为 EndToEndModel
# 请确保 EndToEndModel 与训练时的定义完全一致
model = EndToEndModel().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 加载训练好的权重（这里以 epoch 50 的权重为例，你可以根据需要调整）
checkpoint_path = "weights/2025-03-21-BUSI-18:09:04/model_epoch_200.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

# 设置模型为评估模式
model.eval()

# 定义与训练时相同的预处理流程
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载测试图像和对应的掩码（如果没有掩码可以使用全1掩码）
# test_image_path = "sample/benign.png"  # 替换为你的测试图像路径
test_image_path = "/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/OASBUD/images/1mm.png"  # 替换为你的测试图像路径
test_mask_path = "test_mask.png"    # 替换为你的测试掩码路径

# 读取并预处理图像
image = Image.open(test_image_path).convert("L")
image_tensor = transform(image).unsqueeze(0)  # shape: [1, 1, 256, 256]

# 尝试加载掩码，否则使用全1掩码
try:
    mask = Image.open(test_mask_path).convert("RGB")
    mask_tensor = transform(mask).unsqueeze(0)
except Exception as e:
    mask_tensor = torch.ones_like(image_tensor)

# 移动到设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)
mask_tensor = mask_tensor.to(device)

# 推理（预测）
with torch.no_grad():
    processed_image, segmentation = model(image_tensor, mask_tensor)
    # U-Net 部分输出未经过 Sigmoid，将其映射到 [0,1]
    segmentation = torch.sigmoid(segmentation)  
    segmentation = (segmentation > 0.5).float()

    segmentation = F.interpolate(segmentation, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)
    processed_image = F.interpolate(processed_image, size=image_tensor.shape[2:], mode='bilinear', align_corners=False)


# 将 Tensor 转换为 NumPy 数组用于显示
segmentation_np = segmentation[0].cpu().squeeze().numpy()

processed_image = processed_image[0].cpu().squeeze().numpy()
input_np = image_tensor[0].cpu().squeeze().numpy()

output_folder = "./result"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 以输入图像的文件名保存预测结果
now = datetime.datetime.now()
formatted_time = now.strftime("%Y-%m-%d~%H:%M:%S")
file_name = os.path.basename(test_image_path)

file_name = file_name.split('.')[0] + "_" + formatted_time + "." + file_name.split('.')[1]  # 在文件名后加上时间戳

output_path = os.path.join(output_folder, file_name)


# 显示输入图像和预测的分割结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# plt.title("Input Image")
# plt.imshow(input_np, cmap="gray")

plt.title("Input processed")
plt.imshow(processed_image, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Predicted Segmentation")

plt.imshow(segmentation_np, cmap="gray")
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)




# input_path = "/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/data.mp4"  # 输入路径（图像或视频）


# import cv2
# import torch
# from torchvision import transforms
# from PIL import Image
# import torch.nn.functional as F
# import os
# import numpy as np
# from model import EndToEndModel  # 请确保 EndToEndModel 与训练时的定义一致

# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Running on {device}")

# # 初始化模型，并加载预训练权重
# model = EndToEndModel().to(device)
# checkpoint_path = "weights/2025-03-21 18:09:04/model_epoch_200.pth"
# model.load_state_dict(torch.load(checkpoint_path, map_location=device))
# model.eval()

# # 定义预处理流程（与训练时保持一致）
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

# # 打开视频文件（替换为你的测试视频路径）
# video_path = "/home/ami-1/HUXUFENG/UIstasound/Dataset_BUSI_with_GT/data.mp4"
# cap = cv2.VideoCapture(video_path)
# if not cap.isOpened():
#     print("无法打开视频文件")
#     exit()

# # 设置输出视频参数
# output_folder = "./result"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# fourcc = cv2.VideoWriter_fourcc(*"MJPG")
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# output_path = os.path.join(output_folder, "output_video.avi")
# writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), True)

# print("开始处理视频...")

# frame_count = 0
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     writer.write(frame)
#     frame_count += 1
#     # print(f"Total frames processed: {frame_count}")
#     # 将 BGR 格式转为灰度（根据需要也可以直接用彩色图像）
#     # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # pil_image = Image.fromarray(gray_frame)
#     pil_image= Image.fromarray(frame).convert("L")

#     # 预处理图像：Resize、ToTensor、Normalize
#     image_tensor = transform(pil_image).unsqueeze(0).to(device)  # shape: [1, 1, 256, 256]

#     # 如果没有提供掩码，则使用全1掩码
#     mask_tensor = torch.ones_like(image_tensor).to(device)

#     # 模型推理，生成处理后的图像和分割结果
#     with torch.no_grad():
#         processed_image, segmentation = model(image_tensor, mask_tensor)
#         # U-Net 输出的分割结果未经过 Sigmoid，故需归一化，并二值化
      
#         segmentation = torch.sigmoid(segmentation)
#         segmentation = (segmentation > 0.5).float()

#         # 将输出尺寸调整回原始尺寸（这里调整为 256x256，可根据需要做进一步插值）
#         segmentation = F.interpolate(segmentation, size=(256, 256), mode='bilinear', align_corners=False)
#         processed_image = F.interpolate(processed_image, size=(256, 256), mode='bilinear', align_corners=False)

#     # 将 Tensor 转为 NumPy 数组（乘以255转换为 0-255 的像素值）
#     segmentation_np = segmentation[0].detach().cpu().squeeze().numpy() * 255

#     processed_image_np = processed_image[0].cpu().squeeze().numpy() * 255
#     input_np = image_tensor[0].cpu().squeeze().numpy() * 255

#     segmentation_np = segmentation_np.astype('uint8')
#     processed_image_np = processed_image_np.astype('uint8')
#     input_np = input_np.astype('uint8')

#     # 可选：将分割结果作为掩码叠加到原图上，便于观察分割效果
#     # 这里将灰度图转换为 BGR，以便于后续写入彩色视频
#     input_bgr = cv2.cvtColor(input_np, cv2.COLOR_GRAY2BGR)
#     segmentation_red = np.zeros_like(input_bgr)
#     segmentation_red[:, :, 2] = segmentation_np
#     # segmentation_bgr = cv2.cvtColor(segmentation_np, cv2.COLOR_GRAY2BGR)
#     overlay = cv2.addWeighted(input_bgr, 0.6, segmentation_red, 0.4, 0)

#     # 将处理后的帧写入输出视频
#     writer.write(overlay)

# cap.release()
# writer.release()
# cv2.destroyAllWindows()

# print(f"处理完成，输出视频保存在: {output_path}")

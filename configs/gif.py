import os
import imageio
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import numpy as np

def generate_grid_gif(image_folder, output_path, sample_count=4, duration=1.0):
    """
    将多个 epoch 中的 4 张预测图组合成四宫格，并生成动态GIF。
    
    Args:
        image_folder: 存放所有预测图的路径
        output_path: 保存的 GIF 路径
        sample_count: 每一帧显示几个样本（默认 4 个）
        duration: 每帧显示时间
    """
    # 获取所有 epoch 数
    image_paths = glob(os.path.join(image_folder, "fixed_pred_epoch*_sample1.png"))
    epoch_nums = sorted([
        int(os.path.basename(p).split("epoch")[1].split("_")[0]) for p in image_paths
    ])

    frames = []

    for epoch in epoch_nums:
        fig, axs = plt.subplots(1, sample_count, figsize=(sample_count * 3, 3))
        for i in range(sample_count):
            sample_id = i + 1
            img_path = os.path.join(
                image_folder, f"fixed_pred_epoch{epoch}_sample{sample_id}.png"
            )
            img = Image.open(img_path)
            axs[i].imshow(img, cmap='gray')
            axs[i].set_title(f"Sample {sample_id}")
            axs[i].axis('off')
        fig.suptitle(f"Epoch {epoch}", fontsize=16)

        # 将图保存为 numpy 图像对象
        fig.canvas.draw()
        image_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_np = image_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image_np)
        plt.close(fig)

    # 保存为 GIF
    imageio.mimsave(output_path, frames, duration=duration)
    print(f"[✅] Four-grid GIF saved to: {output_path}")



def generate_individual_gifss(image_folder, output_folder, sample_count=4, duration=5):
    """
    为每个 sample 单独生成一个动态 GIF（显示其随 epoch 变化的预测图）
    """
    os.makedirs(output_folder, exist_ok=True)

    for sample_id in range(1, sample_count + 1):
        pattern = os.path.join(image_folder, f"pred_epoch*_sample{sample_id}.png")
        sample_files = sorted(glob(pattern), key=lambda x: int(x.split("epoch")[1].split("_")[0]))

        if not sample_files:
            print(f"[⚠️] Sample {sample_id} 没有图像，跳过...")
            continue

        frames = []
        base_size = None  # 用于统一尺寸
        for img_path in sample_files:
            img = Image.open(img_path).convert("RGB")
            if base_size is None:
                base_size = img.size  # 设置统一尺寸
            img = img.resize(base_size)  # ⚠️ 关键：统一图像尺寸
            frames.append(np.array(img))

        gif_path = os.path.join(output_folder, f"sample{sample_id}.gif")
        imageio.mimsave(gif_path, frames, duration=duration)
        print(f"[✅] GIF saved: {gif_path}")
import torch
from torchvision.utils import make_grid, save_image

def save_diffusion_comparison(inputs, outputs, masks, save_path, nrow=4):
    """
    拼接原图 + 生成图 + GT Mask，并保存为一张图片。

    Args:
        inputs: 原始输入图像 Tensor，shape (B, 1, H, W)
        outputs: 模型生成图像 Tensor，shape (B, 1, H, W)
        masks: GT 掩膜图像 Tensor，shape (B, 1, H, W)
        save_path: 要保存的路径
        nrow: 每行显示多少张图（默认 4）
    """
    # 反归一化为 [0, 1]，防止图像太暗
    def denorm(x):
        return x.clamp(-1, 1).add(1).div(2)

    inputs = denorm(inputs.detach().cpu())
    outputs = denorm(outputs.detach().cpu())
    masks = denorm(masks.detach().cpu())

    # 拼接格式：[输入 | 输出 | Mask]
    comparison = torch.cat([inputs, outputs, masks], dim=0)  # shape: [3*B, 1, H, W]
    grid = make_grid(comparison, nrow=nrow, pad_value=1.0)

    save_image(grid, save_path)
    print(f"[✅] Comparison image saved to: {save_path}")

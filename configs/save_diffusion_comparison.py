import torch
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt



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
    
    def denorm_s(img):
        return img * 0.5 + 0.5  # 还原到 0-1 范围   

    inputs = denorm_s(inputs.detach().cpu())
    outputs = denorm_s(outputs.detach().cpu())
    masks = denorm_s(masks.detach().cpu())
    

    # 拼接格式：[输入 | 输出 | Mask]
    comparison = torch.cat([inputs, outputs, masks], dim=0)  # shape: [3*B, 1, H, W]
    grid = make_grid(comparison, nrow=nrow, pad_value=1.0)

    save_image(grid, save_path)
    print(f"[✅] Comparison image saved to: {save_path}")


def save_diffusion_comparison_2(inputs, outputs, masks, save_path, nrow=4):
    """
    保存可视化：每个样本拼接 [input | output | mask]，按 nrow 排版保存。

    Args:
        inputs, outputs, masks: (B, 1, H, W)
    """
    def denorm(img):
        return img * 0.5 + 0.5  # 反归一化

    inputs = denorm(inputs.detach().cpu())
    outputs = denorm(outputs.detach().cpu())
    masks = denorm(masks.detach().cpu())

    B = inputs.size(0)
    merged = []

    for i in range(B):
        merged_sample = torch.cat([inputs[i], outputs[i], masks[i]], dim=-1)  # 按宽拼接 H x (W*3)
        merged.append(merged_sample)

    # 合并成一个 batch（B, 1, H, W*3）
    merged = torch.stack(merged)

    # 按行排布输出
    grid = make_grid(merged, nrow=nrow, pad_value=1.0)
    save_image(grid, save_path)
    print(f"[✅] Comparison image saved to: {save_path}")

def save_diffusion_comparison_pro(inputs, outputs, masks, save_path, nrow=4):
    """
    拼接原图 + 生成图 + GT Mask，并保存为一张图片。

    Args:
        inputs: 原始输入图像 Tensor，shape (B, 1, H, W)
        outputs: 模型生成图像 Tensor，shape (B, 1, H, W)
        masks: GT 掩膜图像 Tensor，shape (B, 1, H, W)
        save_path: 保存路径
        nrow: 每行展示几个样本
    """
    def denorm(x):
        return x.clamp(-1, 1).add(1).div(2)  # from [-1, 1] → [0, 1]

    # 确保在 CPU，detach 防止梯度干扰
    inputs = denorm(inputs.detach().cpu())
    outputs = denorm(outputs.detach().cpu())
    masks = denorm(masks.detach().cpu())

    # 拼接格式：[input1, output1, mask1, input2, output2, mask2, ...]
    batch = inputs.shape[0]
    comparisons = []
    for i in range(batch):
        comparisons.extend([inputs[i], outputs[i], masks[i]])  # 每个样本一组三张图

    # make_grid 会自动拼接成网格图像
    grid = make_grid(comparisons, nrow=nrow * 3, pad_value=1.0)

    save_image(grid, save_path)
    print(f"[✅] Comparison image saved to: {save_path}")




def visualize_prediction(image_tensor, mask_tensor, pred_tensor, save_path=None):
    """
    显示输入图像、真实掩码、预测概率图以及预测直方图，并可保存为图片。
    输入都是 torch.Tensor。
    """
    # 转为 numpy，断开梯度追踪
    image_np = image_tensor.squeeze().detach().cpu().numpy()
    mask_np = mask_tensor.squeeze().detach().cpu().numpy()
    pred_np = pred_tensor.squeeze().detach().cpu().numpy()

    plt.figure(figsize=(14, 4))

    # 原始图像
    plt.subplot(1, 4, 1)
    plt.title("Input Image")
    plt.imshow(image_np, cmap='gray')
    plt.axis("off")

    # 真实 mask
    plt.subplot(1, 4, 2)
    plt.title("Ground Truth")
    plt.imshow(mask_np, cmap='gray')
    plt.axis("off")

    # 预测概率图
    plt.subplot(1, 4, 3)
    plt.title("Predicted Prob")
    plt.imshow(pred_np, cmap='gray')
    plt.axis("off")

    # 预测值直方图
    plt.subplot(1, 4, 4)
    plt.title("Pred Value Histogram")
    plt.hist(pred_np.flatten(), bins=50, color='blue', alpha=0.7)
    plt.xlabel("Pred Value")
    plt.ylabel("Count")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        # print(f"[✅] Prediction visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()
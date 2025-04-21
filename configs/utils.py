
import torch
from model import EndToEndModel, ResNet, BasicBlock
import os
from denoising_diffusion import GaussianDiffusion
import glob




# ========== Early Stopper 类 ==========
class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-4, model=None, path=None):
        """
        :param patience: 容忍验证损失无提升的轮数
        :param min_delta: 最小改进幅度
        :param model: 当前训练的模型，用于保存最佳权重
        :param path: 保存权重的目录（非完整文件路径！）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.model = model
        self.path_dir = path
        self.previous_model_path = None  # 上一次保存的权重路径

        # 如果传入路径，确保文件夹存在
        if self.path_dir and not os.path.exists(self.path_dir):
            os.makedirs(self.path_dir, exist_ok=True)

    def __call__(self, current_loss, epoch):
        """
        :param current_loss: 当前验证损失
        :param epoch: 当前训练轮次（用于记录最佳 epoch）
        :return: 是否触发提前停止
        """
        # 当前 loss 更好时保存模型
        if self.best_loss is None or current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0

            # 删除旧的最优权重
            if self.previous_model_path and os.path.exists(self.previous_model_path):
                os.remove(self.previous_model_path)

            # 构建保存路径
            if self.model and self.path_dir:
                filename = f"epoch_{epoch}_best_model.pth" if epoch is not None else "best_model.pth"
                self.previous_model_path = os.path.join(self.path_dir, filename)
                torch.save(self.model.state_dict(), self.previous_model_path)

                # 写入最佳 epoch
                with open(os.path.join(self.path_dir, f"{epoch}"+"Best_epoch.txt"), 'w') as f:
                    f.write(str(epoch) if epoch is not None else "unknown")

                print(f"[📌] Best model saved at epoch {epoch} with val loss {current_loss:.4f}")
        else:
            self.counter += 1

        # 提前停止条件
        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop




def find_latest_diffusion_ckpt(base_dir):
    subdirs = [d for d in os.listdir(base_dir) if "FullPipeline" in d]
    if not subdirs:
        return None
    subdirs.sort(reverse=True)
    latest_dir = os.path.join(base_dir, subdirs[0], "diffusion/best_model")
    print(latest_dir)
    matched = glob.glob(os.path.join(latest_dir, 'epoch_*_best_model.pth'))
    print(matched)
    return matched[0] if matched else None


def get_save_paths(args, sub_module=None):
    assert hasattr(args, 'timestamp'), "args.timestamp must be set in main()"

    mode_to_tag = {
        'diffusion_only': 'Diffusion',
        'unet_only': 'UNet',
        'finetune_unet': 'UNet_Finetune',
        'full_pipeline': 'FullPipeline'
    }
    model_tag = mode_to_tag.get(args.train_mode, 'Unknown')

    # ✅ 使用传入的统一时间戳
    root_dir = os.path.join(args.weights_dir, f"{args.timestamp}_{model_tag}")
    os.makedirs(root_dir, exist_ok=True)

    if args.train_mode == 'full_pipeline' and sub_module in ['diffusion', 'unet']:
        weights_dir = os.path.join(root_dir, sub_module)
        os.makedirs(weights_dir, exist_ok=True)
        csv_file = os.path.join(weights_dir, f"training_{sub_module}_log.csv")
    else:
        weights_dir = root_dir
        csv_file = os.path.join(weights_dir, f"training_{model_tag}_log.csv")

    return weights_dir, csv_file


def load_diffusion_model(args, device):
    model = GaussianDiffusion(
        model=ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1),
        image_size=256,
        timesteps=1000,
        objective='pred_noise',
        beta_schedule='sigmoid',
        auto_normalize=True,
        offset_noise_strength=0.0,
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        immiscible=False
    )
    if args.diffusion_ckpt and os.path.exists(args.diffusion_ckpt):
        model.load_state_dict(torch.load(args.diffusion_ckpt, map_location=device))
        print("[INFO] Loaded diffusion model from:", args.diffusion_ckpt)
    return model.to(device)
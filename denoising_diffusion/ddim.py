import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
# 定义DDIM扩散模型
class DDIMDiffusion(nn.Module):
    def __init__(self, model, image_size, timesteps=1000, sampling_timesteps=None, ddim_sampling_eta=0.0, device=None):
        super(DDIMDiffusion, self).__init__()

        # 设置设备
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps or timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.image_size = image_size

        # 生成beta值和alphas值
        betas = torch.linspace(0.0001, 0.02, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 缓存必要的中间变量
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        # 初始化模型
        self.model = model

    def p_mean_variance(self, x_t, t, x_self_cond=None):
        # 预测从x_t到x_0的概率分布均值与方差
        model_out = self.model(x_t, t, x_self_cond)
        
        model_mean = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x_t + \
                     self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * model_out
        
        # 计算方差（与DDIM不同的是，这里我们不加入噪声）
        posterior_variance = self.betas[t].view(-1, 1, 1, 1)
        return model_mean, posterior_variance

    @torch.no_grad()
    # 修改后的ddim_sample方法
    def ddim_sample(self, shape, eta=None):
        shape = tuple(shape)  # 确保 shape 是一个元组
        
        batch, device = shape[0], self.device
        eta = eta or self.ddim_sampling_eta


        # 随机初始化噪声
        img = torch.randn(*shape, device=device)  # 使用元组解包方式
        imgs = [img]

        # 反向传播过程
        for t in tqdm(reversed(range(self.sampling_timesteps)), desc="DDIM Sampling", total=self.sampling_timesteps):
            time_cond = torch.full((batch,), t, device=device, dtype=torch.long)
            x_start = None
            
            # 计算均值和方差
            model_mean, model_log_variance = self.p_mean_variance(img, time_cond, x_start)
            
            # 计算采样过程中的噪声
            noise = torch.randn_like(img)
            eta_term = eta * ((1 - model_mean ** 2) * noise)
            
            # 更新图像
            img = model_mean + eta_term
            imgs.append(img)

        return img

    def forward(self, x):
        return self.ddim_sample(x)

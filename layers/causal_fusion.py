import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class CausalEncoder(nn.Module):
    """因果编码器：分离混淆因子并进行去混淆编码"""
    def __init__(self, d_in: int, d_latent: int, d_confounder: int, dropout: float = 0.1):
        super().__init__()
        self.d_latent = d_latent
        self.d_confounder = d_confounder
        
        self.confounder_net = nn.Sequential(
            nn.Linear(d_in, d_confounder * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_confounder * 2, d_confounder),
            nn.Tanh()
        )
        
        self.causal_net = nn.Sequential(
            nn.Linear(d_in + d_confounder, d_latent * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent * 2, d_latent)
        )
        
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confounder = self.confounder_net(h)
        h_concat = torch.cat([h, confounder], dim=-1)
        z_causal = self.causal_net(h_concat)
        return z_causal, confounder


class CausalMIEstimator(nn.Module):
    """基于MINE的因果互信息估计器"""
    def __init__(self, d_latent: int, hidden_dim: int = 256):
        super().__init__()
        self.statistics_net = nn.Sequential(
            nn.Linear(d_latent * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, z_t: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        batch_size = z_t.size(0)
        joint = torch.cat([z_t, z_s], dim=-1)
        T_joint = self.statistics_net(joint)
        z_s_shuffled = z_s[torch.randperm(batch_size)]
        marginal = torch.cat([z_t, z_s_shuffled], dim=-1)
        T_marginal = self.statistics_net(marginal)
        
        # MINE下界
        mi_lower_bound = T_joint.mean() - torch.logsumexp(T_marginal, dim=0) + np.log(batch_size)
        mi_lower_bound = mi_lower_bound.mean()  # <-- 确保标量
        return mi_lower_bound


class CausalInterventionModule(nn.Module):
    """因果干预模块"""
    def __init__(self, d_latent: int):
        super().__init__()
        self.intervention_generator = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2),
            nn.ReLU(),
            nn.Linear(d_latent * 2, d_latent)
        )
        self.causal_predictor = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2),
            nn.ReLU(),
            nn.Linear(d_latent * 2, d_latent * 2)
        )
        
    def forward(self, z_t: torch.Tensor, intervention_strength: float = 1.0):
        intervention = self.intervention_generator(z_t)
        z_t_intervened = z_t + intervention_strength * intervention
        causal_params = self.causal_predictor(z_t_intervened)
        mu_causal, log_var_causal = torch.chunk(causal_params, 2, dim=-1)
        return mu_causal, log_var_causal, z_t_intervened


class CounterfactualPredictor(nn.Module):
    """反事实预测器"""
    def __init__(self, d_latent: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_latent, d_latent * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_latent * 2, d_latent)
        )
        
    def forward(self, z_t_cf: torch.Tensor) -> torch.Tensor:
        return self.predictor(z_t_cf)


class LatentProjector(nn.Module):
    """潜在空间投影器"""
    def __init__(self, d_latent: int):
        super().__init__()
        self.mu_proj = nn.Linear(d_latent, d_latent)
        self.logvar_proj = nn.Linear(d_latent, d_latent)
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_proj(z)
        log_var = self.logvar_proj(z)
        return mu, log_var

class CausalDualStreamModel(nn.Module):
    def __init__(self, d_t, d_s, d_latent, d_confounder=32, dropout=0.1):
        super().__init__()
        self.causal_encoder_t = CausalEncoder(d_t, d_latent, d_confounder, dropout)
        self.causal_encoder_s = CausalEncoder(d_s, d_latent, d_confounder, dropout)
        self.mi_estimator = CausalMIEstimator(d_latent)
        self.intervention_module = CausalInterventionModule(d_latent)
        self.counterfactual_predictor = CounterfactualPredictor(d_latent)
        self.latent_proj_t = LatentProjector(d_latent)
        self.latent_proj_s = LatentProjector(d_latent)

    def forward(self, h_t, h_s, 
                add_noise=True, 
                intervention_strength=1.0, 
                counterfactual_noise_std=0.5):
        """
        add_noise: True 表示训练模式，包含采样噪声；False 表示推理模式，使用均值
        """
        # 1. 因果编码
        z_t, conf_t = self.causal_encoder_t(h_t)
        z_s, conf_s = self.causal_encoder_s(h_s)

        # 2. 潜在空间投影
        mu_t, log_var_t = self.latent_proj_t(z_t)
        mu_s, log_var_s = self.latent_proj_s(z_s)

        if add_noise:
            # 重参数化采样
            std_t = torch.exp(0.5 * log_var_t)
            std_s = torch.exp(0.5 * log_var_s)
            eps_t = torch.randn_like(std_t)
            eps_s = torch.randn_like(std_s)
            z_t_sample = mu_t + std_t * eps_t
            z_s_sample = mu_s + std_s * eps_s

            # 因果干预
            mu_causal, log_var_causal, z_t_intervened = self.intervention_module(
                z_t_sample, intervention_strength
            )

            # 反事实生成
            z_t_cf = z_t_sample + torch.randn_like(z_t_sample) * counterfactual_noise_std
            z_s_cf_pred = self.counterfactual_predictor(z_t_cf)
        else:
            # 推理模式：直接使用均值
            z_t_sample = mu_t
            z_s_sample = mu_s
            mu_causal = log_var_causal = z_t_intervened = None
            z_t_cf = z_s_cf_pred = None

        # 融合表示
        z_combined = z_t_sample + z_s_sample

        return {
            'z_t': z_t_sample,
            'z_s': z_s_sample,
            'z_combined': z_combined,
            'mu_t': mu_t,
            'log_var_t': log_var_t,
            'mu_s': mu_s,
            'log_var_s': log_var_s,
            'conf_t': conf_t,
            'conf_s': conf_s,
            'mu_causal': mu_causal,
            'log_var_causal': log_var_causal,
            'z_t_intervened': z_t_intervened,
            'z_t_cf': z_t_cf,
            'z_s_cf_pred': z_s_cf_pred
        }

def hsic_loss(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    batch_size = x.size(0)
    def rbf_kernel(X, sigma):
        X_norm = (X ** 2).sum(1).view(-1, 1)
        return torch.exp(-((X_norm + X_norm.t() - 2.0 * X @ X.t()) / (2 * sigma ** 2)))
    x_flat = x.view(batch_size, -1)
    y_flat = y.view(batch_size, -1)
    K = rbf_kernel(x_flat, sigma)
    L = rbf_kernel(y_flat, sigma)
    H = torch.eye(batch_size, device=x.device) - torch.ones(batch_size, batch_size, device=x.device) / batch_size
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    hsic = torch.trace(K_centered @ L_centered) / ((batch_size - 1) ** 2)
    return hsic


def compute_causal_loss(model: CausalDualStreamModel,
                        h_t: torch.Tensor,
                        h_s: torch.Tensor,
                        lambda_config: Dict[str, float]):
    outputs = model(h_t, h_s)
    L_causal_MI = -model.mi_estimator(outputs['z_t'], outputs['z_s'])
    L_intervention = F.mse_loss(outputs['mu_causal'], outputs['z_s'], reduction='mean')
    noise = torch.randn_like(h_s) * 0.3
    z_s_cf_true, _ = model.causal_encoder_s(h_s + noise)
    L_counterfactual = F.mse_loss(outputs['z_s_cf_pred'], z_s_cf_true, reduction='mean')
    L_confounder = hsic_loss(outputs['conf_t'], outputs['z_t']) + hsic_loss(outputs['conf_s'], outputs['z_s'])
    def kl_divergence(mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
    L_KL = kl_divergence(outputs['mu_t'], outputs['log_var_t']) + kl_divergence(outputs['mu_s'], outputs['log_var_s'])
    L_total = (L_causal_MI +
               lambda_config['intervention'] * L_intervention +
               lambda_config['counterfactual'] * L_counterfactual +
               lambda_config['confounder'] * L_confounder +
               lambda_config['kl'] * L_KL)
    loss_dict = {
        'total': L_total.item(),
        'causal_mi': L_causal_MI.item(),
        'intervention': L_intervention.item(),
        'counterfactual': L_counterfactual.item(),
        'confounder': L_confounder.item(),
        'kl': L_KL.item()
    }
    return L_total, loss_dict


def train_step(model, optimizer, h_t, h_s, lambda_config):
    model.train()
    optimizer.zero_grad()
    loss, loss_dict = compute_causal_loss(model, h_t, h_s, lambda_config)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss_dict

# =================== 使用示例 ===================
if __name__ == "__main__":
    batch_size = 32
    seq_len = 10
    d_t = 64
    d_s = 64
    d_latent = 128
    model = CausalDualStreamModel(d_t, d_s, d_latent, d_confounder=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lambda_config = {'intervention': 0.5, 'counterfactual': 0.3, 'confounder': 0.2, 'kl': 1.0}
    h_t = torch.randn(batch_size, seq_len, d_t)
    h_s = torch.randn(batch_size, seq_len, d_s)

    # 单步训练
    loss_dict = train_step(model, optimizer, h_t, h_s, lambda_config)
    print("训练损失:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    # 推理
    model.eval()
    with torch.no_grad():
        z_combined = model(h_t, h_s, add_noise=False)
        print(f"\n融合表示形状: {z_combined.shape}")

import torch
import torch.nn as nn
import torch.nn.functional as F

class MutualInfoModel(nn.Module):
    def __init__(self, 
                 time_feature_size, 
                 semantic_feature_size,
                 seq_len=10, 
                 latent_size=128, 
                 output_size=21,
                 dropout_rate=0.3):
        super().__init__()
        
        # 时间特征处理
        self.time_encoder = nn.Sequential(
            nn.Linear(time_feature_size, latent_size),
            #nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 语义特征处理
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_feature_size, latent_size),
            #nn.BatchNorm1d(latent_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 潜在空间映射
        self.latent_mu = nn.Linear(latent_size, latent_size)
        self.latent_logvar = nn.Linear(latent_size, latent_size)
        
        # 预测头, 这里的10是时间步数，如何根据不同的时间步数进行调整？
        self.predictor = nn.Sequential(
            nn.Linear(2 * latent_size * seq_len, latent_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(latent_size, output_size)
        )
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧，确保在正确设备上"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def kl_divergence(self, mu, logvar):
        """计算 KL 散度损失"""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    def forward(self, time_feature, semantic_feature):
        # 编码
        time_encoded = self.time_encoder(time_feature)
        semantic_encoded = self.semantic_encoder(semantic_feature)
        
        # 生成潜在空间
        mu_time = self.latent_mu(time_encoded)
        logvar_time = self.latent_logvar(time_encoded)
        mu_semantic = self.latent_mu(semantic_encoded)
        logvar_semantic = self.latent_logvar(semantic_encoded)
        
        # 重参数化
        z_time = self.reparameterize(mu_time, logvar_time)
        z_semantic = self.reparameterize(mu_semantic, logvar_semantic)
        
        # 计算 KL 散度损失
        kl_loss_time = self.kl_divergence(mu_time, logvar_time)
        kl_loss_semantic = self.kl_divergence(mu_semantic, logvar_semantic)
        
        # 潜在空间对比损失（互信息正则化）
        latent_alignment_loss = F.mse_loss(z_time, z_semantic)

        # 展开时间和语义特征：将时间序列展平成一个长向量
        # z_time_flat = z_time.view(z_time.size(0), -1)  # (batch, time_steps * latent_size)
        # z_semantic_flat = z_semantic.view(z_semantic.size(0), -1)  # (batch, time_steps * latent_size)
        
        # 合并潜在表示
        # z_combined = torch.cat([z_time, z_semantic], dim=-1)
        # 加
        z_combined = z_time + z_semantic
        
        # 预测
        # predicted_output = self.predictor(z_combined)

        kl_loss = kl_loss_time + kl_loss_semantic
        
        # # 总损失
        # total_loss = latent_alignment_loss + kl_loss_time + kl_loss_semantic
        
        return latent_alignment_loss, kl_loss, z_combined

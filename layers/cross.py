import torch
import torch.nn as nn
import torch.nn.functional as F

# MultiHeadAttention 模块
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, q, k, v):
        B, Nq, C = q.shape
        Nk = k.shape[1]

        q = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        return self.proj(x)

# TemporalSemanticInteraction 模块
class TemporalSemanticInteraction(nn.Module):
    def __init__(self, llm_dim, dim, num_heads=4, interaction_type='dual'):
        super().__init__()
        self.interaction_type = interaction_type

        # 映射层（可以修改维度映射）
        self.llm_dim = nn.Linear(llm_dim, dim) 

        if interaction_type == 'dual':
            self.cross_attn1 = MultiHeadAttention(dim, num_heads)
            self.cross_attn2 = MultiHeadAttention(dim, num_heads)
        else:  # 协同注意力模式
            self.self_attn = MultiHeadAttention(dim, num_heads)

        self.norm = nn.LayerNorm(dim)
        self.proj_z = nn.Linear(dim, dim)
        self.proj_m = nn.Linear(dim, dim)
    def forward(self, Z_v, M):
        Z_v = self.llm_dim(Z_v)
        assert Z_v.shape[-1] == M.shape[-1], "Dimension mismatch between Z_v and M."
        if self.interaction_type == 'dual':
            # 双向交叉注意力
            Z_v_bar = self.cross_attn1(Z_v, M, M)
            M_bar = self.cross_attn2(M, Z_v, Z_v)
        else:
            # 协同注意力
            combined = torch.cat([Z_v, M], dim=1)
            attended = self.self_attn(combined, combined, combined)
            Z_v_bar, M_bar = torch.split(attended, [Z_v.shape[1], M.shape[1]], dim=1)

        # 残差连接和 LayerNorm
        Z_v_hat = self.norm(self.proj_z(Z_v_bar) + Z_v)
        M_hat = self.norm(self.proj_m(M_bar) + M)

        # 输出拼接
        output = torch.cat([Z_v_hat, M_hat], dim=1)

        return Z_v_hat, M_hat, output

    def compute_regularization_loss(self, M_hat, batch_size, num_slots):
        # 计算视图间的相似分布
        M_hat = M_hat.view(batch_size, num_slots, -1)
        sim_matrix = torch.matmul(M_hat, M_hat.transpose(-2, -1))
        sim_matrix = F.softmax(sim_matrix, dim=-1)

        # 计算正则化损失
        reg_loss = -torch.log(torch.diagonal(sim_matrix, dim1=-2, dim2=-1) + 1e-8).mean()
        return reg_loss
    
if __name__ == '__main__':
    # 测试 TemporalSemanticInteraction
    llm_dim = 128
    dim = 256
    num_heads = 4
    interaction_type = 'dual'
    batch_size = 32
    num_slots = 10
    Z_v = torch.randn(batch_size, num_slots, llm_dim)
    M = torch.randn(batch_size, num_slots, dim)
    tsi = TemporalSemanticInteraction(llm_dim, dim, num_heads, interaction_type)
   
    # 打印可训练参数
    print("模型参数量: ", sum(param.numel() for param in tsi.parameters()))
    Z_v_hat, M_hat, output = tsi(Z_v, M)
    print(output.shape)  # torch.Size([32, 20, 256])
    
    # 测试 compute_regularization_loss
    reg_loss = tsi.compute_regularization_loss(M_hat, batch_size, num_slots)
    print(reg_loss)  # tensor(3.1042, grad_fn=<NegBackward>)
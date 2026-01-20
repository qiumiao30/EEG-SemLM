import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GraphAttentionLayer(nn.Module):
    """
    图注意力层的实现
    输入: (batch_size, seq_len, num_features)
    输出: (batch_size, seq_len, out_features)
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # 定义可训练参数
        # W: 输入变换矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # a: 注意力机制的参数
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # 定义leakyrelu激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x):
        """
        x: 输入特征 (batch_size, seq_len, in_features)
        """
        batch_size, seq_len, _ = x.size()
        
        # 1. 线性变换
        # Wh: (batch_size, seq_len, out_features)
        Wh = torch.matmul(x, self.W)
        
        # 2. 计算注意力分数
        # 为每对节点计算注意力得分
        # 首先准备拼接的特征
        a_input = torch.cat([Wh.repeat_interleave(seq_len, dim=1),
                           Wh.repeat(1, seq_len, 1)], dim=2)
        # reshape成 (batch_size, seq_len * seq_len, 2*out_features)
        a_input = a_input.view(batch_size, seq_len * seq_len, 2 * self.out_features)
        
        # 计算注意力系数 e: (batch_size, seq_len * seq_len, 1)
        e = self.leakyrelu(torch.matmul(a_input, self.a))
        # reshape成邻接矩阵的形状: (batch_size, seq_len, seq_len)
        attention = e.view(batch_size, seq_len, seq_len)
        
        # 3. 归一化注意力分数
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 4. 聚合邻居特征
        # h_prime: (batch_size, seq_len, out_features)
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime, attention

class TemporalAttention(nn.Module):
    """
    时序注意力层的实现
    """
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 定义查询、键、值的变换矩阵
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim)
        """
        # 计算查询、键、值
        Q = self.W_q(x)  # (batch_size, seq_len, hidden_dim)
        K = self.W_k(x)  # (batch_size, seq_len, hidden_dim)
        V = self.W_v(x)  # (batch_size, seq_len, hidden_dim)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        attention = F.softmax(scores, dim=-1)
        
        # 聚合
        output = torch.matmul(attention, V)
        
        return output, attention

class TemporalGAT(nn.Module):
    """
    结合时序注意力和图注意力的模型
    """
    def __init__(self, in_features, hidden_dim, num_heads=4, dropout=0.6):
        super(TemporalGAT, self).__init__()
        
        # 多头图注意力层
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(in_features, hidden_dim, dropout=dropout)
            for _ in range(num_heads)
        ])
        
        # 时序注意力层
        self.temporal_attention = TemporalAttention(hidden_dim * num_heads)
        
        # 输出层
        self.fc = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: 输入特征 (batch_size, seq_len, in_features)
        """
        # 1. 多头图注意力
        gat_outputs = []
        gat_attentions = []
        for gat_layer in self.gat_layers:
            output, attention = gat_layer(x)
            gat_outputs.append(output)
            gat_attentions.append(attention)
        
        # 连接多头的输出
        gat_output = torch.cat(gat_outputs, dim=-1)
        
        # 2. 时序注意力
        temporal_output, temporal_attention = self.temporal_attention(gat_output)
        
        # 3. 最终输出层
        output = self.fc(self.dropout(temporal_output))
        
        return output, gat_attentions, temporal_attention, temporal_output

# 使用示例
def main():
    # 参数设置
    batch_size = 32
    seq_len = 24
    in_features = 16
    hidden_dim = 128
    
    # 创建模型
    model = TemporalGAT(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_heads=4,
        dropout=0.6
    )
    
    # 创建示例输入
    x = torch.randn(batch_size, seq_len, in_features)
    
    # 前向传播
    output, gat_attentions, temporal_attention, temporal_output = model(x)
    
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("图注意力权重形状:", gat_attentions[0].shape)
    print("时序注意力权重形状:", temporal_attention.shape)
    print("时序注意力输出形状:", temporal_output.shape)

if __name__ == "__main__":
    main()
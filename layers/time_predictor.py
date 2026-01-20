import torch
import torch.nn as nn
import torch.nn.functional as F

class TimePredictor(nn.Module):
    def __init__(self, input_dim=512, output_dim=21, hidden_dim=32, pooling_type='mean'):
        super().__init__()
        
        # 输入投影层 (如果需要)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim > hidden_dim else None
        
        # 时序处理模块 (简单化为单层)
        self.temporal_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 聚合方式
        self.pooling_type = pooling_type
        self.attention = nn.Linear(hidden_dim, 1) if pooling_type == 'attention' else None
        
        # 输出预测层
        self.predictor = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        if self.input_proj:
            x = self.input_proj(x)
        
        # 处理序列特征
        h = self.temporal_block(x)
        
        # 聚合
        if self.pooling_type == 'attention':
            attention_weights = torch.softmax(self.attention(h).squeeze(-1), dim=1)
            h_pooled = torch.sum(h * attention_weights.unsqueeze(-1), dim=1)
        elif self.pooling_type == 'mean':
            h_pooled = h.mean(dim=1)
        elif self.pooling_type == 'max':
            h_pooled = h.max(dim=1)[0]
        else:
            h_pooled = h[:, -1, :]
        
        return self.predictor(h_pooled)


# if __name__ == "__main__":  
#     model = TimePredictor(input_dim=2048, output_dim=51, hidden_dim=128, pooling_type='max')
#     x = torch.randn(128, 1010, 2048)
#     # 打印可训练参数量
#     print("模型参数量: ", sum(param.numel() for param in model.parameters()))
#     import time
#     start = time.time()
#     output = model(x)
#     print("模型运行时间: ", time.time() - start)
#     print(output.shape)  # torch.Size([2, 51])
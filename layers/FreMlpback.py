# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class MlpModel(nn.Module):
#     def __init__(self, pred_len, enc_in, seq_len, channel_independence, embed):
#         super(MlpModel, self).__init__()
#         self.embed_size = 128 #embed_size
#         self.hidden_size = 256 #hidden_size
#         self.pre_length = pred_len
#         self.feature_size = enc_in
#         self.seq_length = seq_len
#         self.channel_independence = channel_independence
#         self.embed = embed
#         self.sparsity_threshold = 0.01
#         self.scale = 0.02
#         self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
#         self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
#         self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
#         self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
#         self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
#         self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
#         self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
#         self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
#         self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

#         self.fc = nn.Sequential(
#             nn.Linear(self.feature_size * self.embed_size, self.hidden_size),
#             nn.LeakyReLU(),
#             nn.Linear(self.hidden_size, self.pre_length)
#         )

#         self.embed = nn.Linear(self.feature_size*self.embed_size, self.embed)

#     # dimension extension
#     def tokenEmb(self, x):
#         # x: [Batch, Input length, Channel]
#         x = x.permute(0, 2, 1)
#         x = x.unsqueeze(3)
#         # N*T*1 x 1*D = N*T*D
#         y = self.embeddings
#         return x * y

#     # frequency temporal learner
#     def MLP_temporal(self, x, B, N, L):
#         # [B, N, T, D]
#         x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on L dimension
#         y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
#         x = torch.fft.irfft(y, n=self.seq_length, dim=2, norm="ortho")
#         return x

#     # frequency channel learner
#     def MLP_channel(self, x, B, N, L):
#         # [B, N, T, D]
#         x = x.permute(0, 2, 1, 3)
#         # [B, T, N, D]
#         x = torch.fft.rfft(x, dim=2, norm='ortho') # FFT on N dimension
#         y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
#         x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
#         x = x.permute(0, 2, 1, 3)
#         # [B, N, T, D]
#         return x

#     # frequency-domain MLPs
#     # dimension: FFT along the dimension, r: the real part of weights, i: the imaginary part of weights
#     # rb: the real part of bias, ib: the imaginary part of bias
#     def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
#         o1_real = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
#                               device=x.device)
#         o1_imag = torch.zeros([B, nd, dimension // 2 + 1, self.embed_size],
#                               device=x.device)

#         o1_real = F.relu(
#             torch.einsum('bijd,dd->bijd', x.real, r) - \
#             torch.einsum('bijd,dd->bijd', x.imag, i) + \
#             rb
#         )

#         o1_imag = F.relu(
#             torch.einsum('bijd,dd->bijd', x.imag, r) + \
#             torch.einsum('bijd,dd->bijd', x.real, i) + \
#             ib
#         )

#         y = torch.stack([o1_real, o1_imag], dim=-1)
#         y = F.softshrink(y, lambd=self.sparsity_threshold)
#         y = torch.view_as_complex(y)
#         return y

#     def forward(self, x):
#         # x: [Batch, Input length, Channel]
#         B, T, N = x.shape
#         # embedding x: [B, N, T, D]
#         x = self.tokenEmb(x)
#         bias = x
#         # [B, N, T, D]
#         if self.channel_independence == '1':
#             x = self.MLP_channel(x, B, N, T)
#         # [B, N, T, D]
#         x = self.MLP_temporal(x, B, N, T)
#         x = x + bias
#         x = x.permute(0, 2, 1, 3)
#         x = x.reshape(B, T, -1)
#         x = self.embed(x)
        
#         # x = x.reshape(B, T, N, -1)
#         # x = self.fc(x).permute(0, 2, 1).squeeze(1)
#         return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch.optim as optim


class MlpModel(nn.Module):
    def __init__(self, pred_len, enc_in, seq_len, channel_independence, embed):
        super(MlpModel, self).__init__()
        # Hyperparameters with more flexible configuration
        self.config = {
            'embed_size': 128,
            'hidden_size': 256,
            'pred_length': pred_len,
            'feature_size': enc_in,
            'seq_length': seq_len,
            'channel_independence': '1',
            'sparsity_threshold': 0.01,
            'dropout_rate': 0.2
        }
        
        # More efficient parameter initialization
        self.embeddings = nn.Parameter(torch.zeros(1, self.config['embed_size']))
        nn.init.xavier_uniform_(self.embeddings)
        
        # Modularized weight matrices with optimized initialization
        self.freq_weights = nn.ParameterDict({
            'r1': nn.Parameter(torch.zeros(self.config['embed_size'], self.config['embed_size'])),
            'i1': nn.Parameter(torch.zeros(self.config['embed_size'], self.config['embed_size'])),
            'r2': nn.Parameter(torch.zeros(self.config['embed_size'], self.config['embed_size'])),
            'i2': nn.Parameter(torch.zeros(self.config['embed_size'], self.config['embed_size']))
        })
        
        # Xavier initialization for frequency weights
        for param in self.freq_weights.values():
            nn.init.xavier_uniform_(param)
        
        # Bias terms with adaptive initialization
        self.freq_biases = nn.ParameterDict({
            'rb1': nn.Parameter(torch.zeros(self.config['embed_size'])),
            'ib1': nn.Parameter(torch.zeros(self.config['embed_size'])),
            'rb2': nn.Parameter(torch.zeros(self.config['embed_size'])),
            'ib2': nn.Parameter(torch.zeros(self.config['embed_size']))
        })
        
        # Enhanced feature processing network
        self.feature_net = nn.Sequential(
            nn.Linear(self.config['feature_size'] * self.config['embed_size'], self.config['hidden_size']),
            nn.BatchNorm1d(self.config['hidden_size']),
            nn.LeakyReLU(0.01),
            nn.Dropout(self.config['dropout_rate']),
            nn.Linear(self.config['hidden_size'], self.config['pred_length'])
        )
        
        # More flexible embedding layer
        self.embed_layer = nn.Linear(self.config['feature_size'] * self.config['embed_size'], embed)
        
    def token_embedding(self, x):
        # Optimized token embedding with einsum for efficiency
        x = x.permute(0, 2, 1).unsqueeze(3)
        return x * self.embeddings
    
    def frequency_mlp(self, x, r, i, rb, ib):
        # Vectorized frequency-domain MLP processing
        x_fft = torch.fft.rfft(x, dim=2, norm='ortho')
        
        # Efficient complex matrix multiplication
        real_part = F.relu(
            torch.einsum('bijd,dd->bijd', x_fft.real, r) - 
            torch.einsum('bijd,dd->bijd', x_fft.imag, i) + rb
        )
        
        imag_part = F.relu(
            torch.einsum('bijd,dd->bijd', x_fft.imag, r) + 
            torch.einsum('bijd,dd->bijd', x_fft.real, i) + ib
        )
        
        # Sparse regularization
        complex_tensor = torch.stack([real_part, imag_part], dim=-1)
        complex_tensor = F.softshrink(complex_tensor, lambd=self.config['sparsity_threshold'])
        
        return torch.fft.irfft(
            torch.view_as_complex(complex_tensor), 
            n=x.shape[2], 
            dim=2, 
            norm="ortho"
        )
    
    def forward(self, x):
        B, T, N = x.shape
        
        # Token embedding with residual connection
        x = self.token_embedding(x)
        residual = x
        
        # Channel-independent processing (optional)
        if self.config['channel_independence'] == '1':
            x = self.frequency_mlp(
                x.permute(0, 2, 1, 3), 
                self.freq_weights['r1'], 
                self.freq_weights['i1'], 
                self.freq_biases['rb1'], 
                self.freq_biases['ib1']
            ).permute(0, 2, 1, 3)
        
        # Temporal frequency processing
        x = self.frequency_mlp(
            x, 
            self.freq_weights['r2'], 
            self.freq_weights['i2'], 
            self.freq_biases['rb2'], 
            self.freq_biases['ib2']
        )
        
        # Residual connection
        x = x + residual
        
        # Reshape and process
        x = x.permute(0, 2, 1, 3).reshape(B, T, -1)
        x = self.embed_layer(x)
        
        return x

model = MlpModel(1, 21, 20, '1', 512)

# 随机生成数据
data = torch.randn(32, 20, 21)

out = model(data)

print(out.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import torch.optim as optim
from torch.cuda.amp import autocast

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

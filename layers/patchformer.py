import torch
from torch import nn
import torch.nn.functional as F
import math

# Self Attention Class
class SelfAttentionConv(nn.Module):
    def __init__(self, k, headers, seq_length, kernel_size=5, mask_next=True, mask_diag=False):
        super().__init__()

        self.k, self.headers, self.kernel_size = k, headers, kernel_size
        self.mask_next = mask_next
        self.mask_diag = mask_diag

        h = headers

        # Query, Key and Value Transformations
        padding = (kernel_size - 1) * 2
        self.padding_operator = nn.ConstantPad1d((padding, 0), 0)  # (padding, 0) --> (左填充，右填充)

        self.toqueries = nn.Conv1d(k, k * h, kernel_size, padding=0, bias=True, dilation=2)  # (k=in_channel, k*h=out_channel)
        self.tokeys = nn.Conv1d(k, k * h, kernel_size, padding=0, bias=True, dilation=2)
        self.tovalues = nn.Conv1d(k, k * h, kernel_size=1, padding=0, bias=False)  # No convolution operated

        self.linear = nn.Conv1d(k, k * h, kernel_size=1, padding=0, bias=False)
        
        position_ids_l = torch.arange(seq_length, dtype=torch.long).view(-1, 1)
        position_ids_r = torch.arange(seq_length, dtype=torch.long).view(1, -1)
        self.distance = (position_ids_r - position_ids_l).abs()
        self.window_size = 5

        self.dropout = nn.Dropout(p=0.2)
        self.silu = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.gate.data.fill_(1)

        # Heads unifier
        self.unifyheads = nn.Linear(k * h, k)

    def forward(self, x):
        # Extraction dimensions
        b, t, k = x.size()  # batch_size, number_of_timesteps, number_of_time_series

        # Checking Embedding dimension
        assert self.k == k, 'Number of time series ' + str(k) + ' didn’t match the number of k ' + str(self.k) + ' in the initialization of the attention layer.'
        h = self.headers

        # Transpose to see the different time series as different channels
        x = x.transpose(1, 2)
        x_padded = self.padding_operator(x)

        # Query, Key and Value Transformations
        queries_g = self.silu(self.toqueries(x_padded).view(b, k, h, t))
        keys_g = self.silu(self.tokeys(x_padded).view(b, k, h, t))
        values_g = self.silu(self.tovalues(x).view(b, k, h, t))

        # Transposition to return the canonical format
        queries_g = queries_g.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        queries_g = queries_g.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        values_g = values_g.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        values_g = values_g.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        keys_g = keys_g.transpose(1, 2)  # batch, header, time serie, time step (b, h, k, t)
        keys_g = keys_g.transpose(2, 3)  # batch, header, time step, time serie (b, h, t, k)

        # Weights
        queries_g = queries_g / (k ** (.25))
        keys_g = keys_g / (k ** (.25))

        queries_g = queries_g.transpose(1, 2).contiguous().view(b * h, t, k)
        keys_g = keys_g.transpose(1, 2).contiguous().view(b * h, t, k)
        values_g = values_g.transpose(1, 2).contiguous().view(b * h, t, k)

        weights = torch.bmm(queries_g, keys_g.transpose(1, 2))

        # Mask the upper & diag of the attention matrix
        if self.mask_next:
            if self.mask_diag:
                indices = torch.triu_indices(t, t, offset=0)
                weights[:, indices[0], indices[1]] = float('-inf')
                # scores_l[:, indices[0], indices[1]] = float('-inf')
            else:
                indices = torch.triu_indices(t, t, offset=1)
                weights[:, indices[0], indices[1]] = float('-inf')
                # scores_l[:, indices[0], indices[1]] = float('-inf')

        # Softmax
        # p_attn_l = self.dropout(F.softmax(scores_l, dim=-1))

        weights = F.softmax(weights, dim=2)

        # Output
        output_g = torch.bmm(weights, values_g)
        # output_l = torch.matmul(p_attn_l, values_l)

        output_g = output_g.view(b, h, t, k)
        # output_l = output_l.view(b, h, t, k)

        # output = torch.cat([output_l, output_g], dim=1)
        output = output_g.transpose(1, 2).contiguous().view(b, t, k * h)

        return self.silu(self.unifyheads(output))  # shape (b,t,k)


# Conv Transformer Block
class ConvTransformerBlock(nn.Module):
    def __init__(self, k, headers, seq_length, kernel_size=5, mask_next=True, mask_diag=False, dropout_proba=0.2):
        super().__init__()

        # Self attention
        self.attention = SelfAttentionConv(k, headers, seq_length, kernel_size, mask_next, mask_diag)

        # First & Second Norm
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        # Feed Forward Network
        self.feedforward = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )
        
        # Dropout & Relu
        self.dropout = nn.Dropout(p=dropout_proba)
        self.activation = nn.ReLU()

    def forward(self, x, train=False):
        # Self attention + Residual
        x = self.attention(x) + x

        # Dropout attention
        if train:
            x = self.dropout(x)

        # First Normalization
        x = self.norm1(x)

        # Feed Forward network + residual
        x = self.feedforward(x) + x

        # Second Normalization
        x = self.norm2(x)

        return x


# Forecasting Conv Transformer
class ForcastConvTransformer(nn.Module):
    def __init__(self, k, features, headers, depth, seq_length, kernel_size=5, mask_next=True, mask_diag=False,
                 dropout_proba=0.2, num_tokens=None):
        super().__init__()
        
        # Embedding
        self.tokens_in_count = False
        if num_tokens:
            self.tokens_in_count = True
            self.token_embedding = nn.Embedding(num_tokens, k)  # seq_length : windows

        # Position embedding
        self.position_embedding = nn.Embedding(seq_length, k)

        # Number of time series
        self.k = k  # num_of_var
        self.seq_length = seq_length

        # Transformer blocks
        tblocks = []
        for t in range(depth):
            tblocks.append(ConvTransformerBlock(k, headers, seq_length, kernel_size, mask_next, mask_diag, dropout_proba))
        self.TransformerBlocks = nn.Sequential(*tblocks)

        self.Linear = nn.Linear(k, features)

    def forward(self, x, tokens=None):
        b, t, k = x.size()

        # Position embedding
        pos = torch.arange(t).to(x.device)  # Ensure position tensor is on the same device
        self.pos_emb = self.position_embedding(pos).unsqueeze(0).expand(b, t, k)
       
        x_ori = self.pos_emb + x

        hidden = self.TransformerBlocks(x_ori)

        hidden = self.Linear(hidden)

        return hidden
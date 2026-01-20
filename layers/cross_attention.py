# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CrossAttentionFusion(nn.Module):
#     def __init__(self, hidden_size_llm: int = 2048, hidden_size_time_series: int = 512, num_heads: int = 2):
#         super(CrossAttentionFusion, self).__init__()
        
#         # 线性投影层：将 LLM 和时序特征映射到相同的维度
#         self.proj_llm = nn.Linear(hidden_size_llm, hidden_size_time_series)  # 将 LLM 特征投影到 512 维
#         self.proj_ts = nn.Linear(hidden_size_time_series, hidden_size_llm)  # 将时序特征投影到 2048 维
        
#         # Cross-attention for fusion
#         self.attn_llm_to_ts = nn.MultiheadAttention(embed_dim=hidden_size_time_series, num_heads=num_heads, batch_first=True)
#         self.self_attn_llm = nn.MultiheadAttention(embed_dim=hidden_size_llm, num_heads=num_heads, batch_first=True)
#         self.attn_ts_to_llm = nn.MultiheadAttention(embed_dim=hidden_size_llm, num_heads=num_heads, batch_first=True)

        
#         # Self-attention layers for intra-modality information fusion
#         self.self_attn_ts = nn.MultiheadAttention(embed_dim=hidden_size_time_series, num_heads=num_heads, batch_first=True)
#         # Dynamic weight generation layers
#         self.weight_gen_llm = nn.Linear(hidden_size_llm, 1)
#         self.weight_gen_ts = nn.Linear(hidden_size_time_series, 1)
        
#         # Non-linear fusion layers
#         self.fc1 = nn.Linear(hidden_size_llm + hidden_size_time_series, hidden_size_llm)  # Fusion projection
#         self.fc2 = nn.Linear(hidden_size_llm, hidden_size_llm)  # Output projection
#         self.relu = nn.ReLU()  # Activation function
        
#         # Optional: Dropout to prevent overfitting
#         self.dropout = nn.Dropout(p=0.2)

#     def forward(self, llm_hidden: torch.Tensor, time_series_hidden: torch.Tensor):
#         """
#         llm_hidden: (B, 3000, 2048) - LLM-based features
#         time_series_hidden: (B, 20, 512) - Time-series features
#         """

#         # Step 1: Cross-attention from LLM to Time-Series
#         # 先将 LLM 特征投影到与时序特征相同的维度
#         projected_llm = self.proj_llm(llm_hidden)  # (B, token, 512)
#         # projected_llm = projected_llm.to(torch.float16)
        
#         # Use projected LLM features as queries and Time-series features as keys/values
#         attn_output_llm_to_ts, _ = self.attn_llm_to_ts(projected_llm, time_series_hidden, time_series_hidden)  # (B, token, 512)
        
#         # Step 2: Cross-attention from Time-Series to LLM
#         # 先将时序特征投影到与 LLM 特征相同的维度
#         projected_ts = self.proj_ts(time_series_hidden)  # (B, series, 2048)
#         # projected_ts = projected_ts.to(torch.float16)
        
#         # Use projected Time-series features as queries and LLM features as keys/values
#         attn_output_ts_to_llm, _ = self.attn_ts_to_llm(projected_ts, llm_hidden, llm_hidden)  # (B, series, 2048)
        
#         # Step 3: Intra-modality self-attention for each modality
#         self_attn_llm, _ = self.self_attn_llm(llm_hidden, llm_hidden, llm_hidden)  # (B, token, 2048)
#         self_attn_ts, _ = self.self_attn_ts(time_series_hidden, time_series_hidden, time_series_hidden)  # (B, series, 512)
        
#         # Step 4: Fuse the outputs (cross-attention + self-attention)
#         # First concatenate cross-attention and self-attention outputs for each modality
#         fused_llm = torch.cat([attn_output_llm_to_ts, self_attn_llm], dim=-1)  # (B, token, 2560)
#         fused_ts = torch.cat([attn_output_ts_to_llm, self_attn_ts], dim=-1)  # (B, series, 2560)

#         # Step 5: Compute dynamic weights for each modality
#         # weight_llm = self.weight_gen_llm(fused_llm.mean(dim=1))  # Global attention to generate weight for LLM
#         # weight_ts = self.weight_gen_ts(fused_ts.mean(dim=1))    # Global attention to generate weight for Time-Series
        
#         # # Step 6: Normalize the weights
#         # weight_llm = torch.sigmoid(weight_llm)  # Sigmoid to constrain between 0 and 1
#         # weight_ts = torch.sigmoid(weight_ts)    # Sigmoid to constrain between 0 and 1
        
#         # # Step 7: Apply dynamic weights to the fused output
#         # weighted_fused_output = weight_llm.unsqueeze(-1) * fused_llm + weight_ts.unsqueeze(-1) * fused_ts

#         fused_output = torch.cat([fused_llm, fused_ts], dim=1)
        
#         # Step 8: Apply non-linear fusion layers
#         fused_output = self.fc1(fused_output)  # Linear transformation
#         fused_output = self.relu(fused_output)  # Non-linear activation
#         fused_output = self.dropout(fused_output)  # Optional Dropout for regularization

#         # Step 9: Further projection to desired output size
#         output = self.fc2(fused_output)  # Final output projection

#         return output


import torch
import torch.nn as nn
import torch.nn.functional as F


class BiDirectionalCrossAttention(nn.Module):
    def __init__(self, latent_dim, token_dim):
        """
        Parameters:
        latent_dim: int, dimension of latent vectors
        token_dim: int, dimension of token embeddings
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.token_dim = token_dim

        # Linear projections
        self.latent_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.token_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value_proj_lat = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value_proj_tok = nn.Linear(latent_dim, latent_dim, bias=False)

        # 基于一维卷积将token的序列长度进行采样
        self.conv1d = nn.Conv1d(in_channels=token_dim, out_channels=latent_dim, kernel_size=5, padding=0, stride=5)

    def forward(self, latents, tokens):
        """
        Parameters:
        latents: Tensor of shape (batch_size, latent_count, latent_dim)
        tokens: Tensor of shape (batch_size, seq_len, token_dim)

        Returns:
        updated_latents: Tensor of shape (batch_size, latent_count, latent_dim)
        updated_tokens: Tensor of shape (batch_size, seq_len, token_dim)
        """
        batch_size, latent_count, latent_dim = latents.shape
        tokens = self.conv1d(tokens.permute(0, 2, 1)).permute(0, 2, 1)
        seq_len, token_dim = tokens.shape[1], tokens.shape[2]

        # Project latents and tokens
        R_lat = self.latent_proj(latents)  # (batch_size, latent_count, latent_dim)
        R_tok = self.token_proj(tokens)  # (batch_size, seq_len, latent_dim)
        V_lat = self.value_proj_lat(latents)  # (batch_size, latent_count, latent_dim)
        V_tok = self.value_proj_tok(tokens)  # (batch_size, seq_len, latent_dim)

        # Compute bi-directional attention
        A_lat_tok = torch.matmul(R_lat, R_tok.transpose(-1, -2)) / self.latent_dim ** 0.5  # (batch_size, latent_count, seq_len)
        A_tok_lat = A_lat_tok.transpose(-1, -2)  # (batch_size, seq_len, latent_count)

        # Compute attention updates
        Delta_lat = F.softmax(A_lat_tok, dim=-1) @ V_tok  # (batch_size, latent_count, latent_dim)
        Delta_tok = F.softmax(A_tok_lat, dim=-1) @ V_lat  # (batch_size, seq_len, latent_dim)

        # Update latents and tokens
        updated_latents = latents + Delta_lat  # (batch_size, latent_count, latent_dim)
        updated_tokens = tokens + Delta_tok  # (batch_size, seq_len, token_dim)

        output = torch.cat([updated_latents, updated_tokens], dim=1)

        return updated_latents, updated_tokens, output
    
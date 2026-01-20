import torch
import torch.nn as nn
from einops import rearrange
from transformers import GPT2Model, GPT2Tokenizer

# --- 导入你的因果融合模块 ---
from causal_fusion import CausalDualStreamModel, compute_causal_loss

# --- 假设你之前的 ForcastConvTransformer 已经存在 ---
from patchformer import ForcastConvTransformer

class EEGAnomalyModel(nn.Module):
    def __init__(self, configs):
        super(EEGAnomalyModel, self).__init__()
        self.task_name = 'anomaly_detection'
        self.d_ff = configs['d_ff']         # Feed-forward dimension (e.g., 512)
        self.seq_len = configs['seq_len']   # Sequence length (e.g., 10)
        self.pred_len = configs['pred_len'] # Prediction length (e.g., 1)

        # GPT2 for dynamic prompt embeddings
        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2.h = self.gpt2.h[:configs['gpt_layers']]

        # Freeze GPT2 weights if specified
        for param in self.gpt2.parameters():
            param.requires_grad = configs.get('train_gpt2', False)

        # Output layer
        self.out_layer = nn.Linear(self.d_ff, configs['c_out'], bias=True)

        # Temporal feature extractor
        self.temporal = ForcastConvTransformer(
            configs['enc_in'], configs['d_model'], configs['n_heads'],
            configs['n_layers'], configs['seq_len'], kernel_size=3,
            mask_next=True, mask_diag=False, dropout_proba=0.2
        )

        # Replace fusion with CausalDualStreamModel
        self.fusion = CausalDualStreamModel(
            d_t=configs['d_model'],
            d_s=configs['d_model'],
            d_latent=configs['d_model'],
            d_confounder=32,
            dropout=0.1
        )

    def construct_dynamic_prompt(self, x_enc):
        means = x_enc.mean(dim=1)
        stdev = torch.std(x_enc, dim=1)
        max_vals = x_enc.max(dim=1).values
        min_vals = x_enc.min(dim=1).values

        mean_value = means.mean().item()
        stdev_value = stdev.mean().item()
        max_value = max_vals.mean().item()
        min_value = min_vals.mean().item()

        prompt = (f"EEG anomaly detection. Signal: avg={mean_value:.2f}, std={stdev_value:.2f}, "
                  f"range=({min_value:.2f}, {max_value:.2f}). "
                  "EEG bands: Alpha (8-13 Hz), Beta (13-30 Hz), Delta (0.5-4 Hz), Theta (4-8 Hz). "
                  "Focus on detecting spikes, sudden amplitude shifts, or unusual frequency patterns. "
                  "Anomalies may include epileptic spikes, seizures, or sleep-related patterns, "
                  "often reflected in abnormal signal changes or extreme deviations in specific channels.")
        return prompt

    def forward(self, x_enc):
        B, L, M = x_enc.shape
        seg_num = 2
        if L % seg_num != 0:
            raise ValueError(f"Sequence length {L} should be divisible by seg_num {seg_num}.")

        # --- Segment normalization ---
        x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=seg_num)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')

        # Temporal features
        x_temporal = self.temporal(x_enc)

        # --- GPT2 prompt ---
        padding_size = 768 - x_enc.shape[-1]
        if padding_size > 0:
            x_enc = torch.nn.functional.pad(x_enc, (0, padding_size))

        prompt = self.construct_dynamic_prompt(x_enc)
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt')
        prompt_ids = prompt_tokens['input_ids'].squeeze(0).to(x_enc.device)
        prompt_embeds = self.gpt2.wte(prompt_ids)
        prompt_embeds = prompt_embeds.unsqueeze(0).repeat(B, 1, 1)
        x_enc = torch.cat([prompt_embeds, x_enc], dim=1)

        # GPT2 forward
        outputs = self.gpt2(inputs_embeds=x_enc).last_hidden_state
        outputs = outputs[:, prompt_embeds.shape[1]:, :self.d_ff]

        # --- Causal fusion ---
        lambda_config = {
            'intervention': 0.5,
            'counterfactual': 0.3,
            'confounder': 0.2,
            'kl': 1.0
        }
        fusion_outputs = self.fusion(x_temporal, outputs)
        laten_loss, loss_dict = compute_causal_loss(self.fusion, x_temporal, outputs, lambda_config)
        outputs = fusion_outputs['z_combined']
        kl_loss = loss_dict['kl']

        # --- Output layer ---
        dec_out = self.out_layer(outputs)
        dec_out = rearrange(dec_out, 'b (n s) m -> b n s m', s=seg_num)
        dec_out = dec_out * stdev[:, :, 0, :].unsqueeze(2).repeat(1, 1, seg_num, 1)
        dec_out = dec_out + means[:, :, 0, :].unsqueeze(2).repeat(1, 1, seg_num, 1)
        dec_out = rearrange(dec_out, 'b n s m -> b (n s) m')
        dec_out = dec_out[:, -self.pred_len:, :]

        return laten_loss, kl_loss, dec_out.squeeze(1)


# --- 测试前向 ---
if __name__ == "__main__":
    configs = {
        'd_ff': 128,
        'seq_len': 10,
        'pred_len': 1,
        'gpt_layers': 4,
        'c_out': 1,
        'enc_in': 21,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'train_gpt2': False
    }

    model = EEGAnomalyModel(configs)
    B, L, M = 32, 10, 21
    x_enc = torch.randn(B, L, M)

    laten_loss, kl_loss, dec_out = model(x_enc)
    print(f"latent loss: {laten_loss.item():.4f}, kl_loss: {kl_loss:.4f}, dec_out shape: {dec_out.shape}")

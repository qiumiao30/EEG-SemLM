import torch
import torch.nn as nn
from einops import rearrange
from transformers import GPT2Model, GPT2Tokenizer
from layers.fusion import MutualInfoModel
from layers.patchformer import ForcastConvTransformer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = 'anomaly_detection'
        self.d_ff = configs['d_ff']         # Feed-forward dimension (e.g., 512)
        self.seq_len = configs['seq_len']   # Sequence length (e.g., 10)
        self.pred_len = configs['pred_len'] # Prediction length (e.g., 1)

        # Initialize GPT2 and tokenizer for embedding textual prompts
        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Tokenizer for GPT-2
        self.gpt2.h = self.gpt2.h[:configs['gpt_layers']]  # Keep only the first N layers based on config

        # Freeze GPT2 weights except for the output layer (optional)
        for param in self.gpt2.parameters():
            param.requires_grad = configs.get('train_gpt2', False)

        # Output layer for anomaly detection
        self.out_layer = nn.Linear(self.d_ff, configs['c_out'], bias=True)

        self.temporal = ForcastConvTransformer(configs['enc_in'], configs['d_model'], configs['n_heads'], configs['n_layers'], configs['seq_len'], 
                                               kernel_size=3, mask_next=True, mask_diag=False, dropout_proba=0.2)

        self.fusion = MutualInfoModel(configs['d_model'], configs['d_model'], seq_len=configs['seq_len'], latent_size=configs['d_model'])

    def construct_dynamic_prompt(self, x_enc):
        """
        Construct a concise and focused prompt for EEG anomaly detection based on input data.
        """
        # Compute basic statistics for x_enc
        means = x_enc.mean(dim=1)  # Mean across sequence dimension
        stdev = torch.std(x_enc, dim=1)  # Standard deviation across sequence dimension
        max_vals = x_enc.max(dim=1).values  # Maximum values along sequence
        min_vals = x_enc.min(dim=1).values  # Minimum values along sequence

        # Calculate mean and std deviation across the batch
        mean_value = means.mean().item()
        stdev_value = stdev.mean().item()
        max_value = max_vals.mean().item()
        min_value = min_vals.mean().item()

        # Construct the prompt
        prompt = (f"EEG anomaly detection. Signal: avg={mean_value:.2f}, std={stdev_value:.2f}, "
                f"range=({min_value:.2f}, {max_value:.2f}). "
                "EEG bands: Alpha (8-13 Hz), Beta (13-30 Hz), Delta (0.5-4 Hz), Theta (4-8 Hz). "
                "Focus on detecting spikes, sudden amplitude shifts, or unusual frequency patterns. "
                "Anomalies may include epileptic spikes, seizures, or sleep-related patterns, "
                "often reflected in abnormal signal changes or extreme deviations in specific channels.")

        return prompt


    def forward(self, x_enc):
        B, L, M = x_enc.shape  # B: batch size, L: sequence length, M: feature dim

        # x_temporal = self.temporal(x_enc)

        # Step 1: Normalization (by segment)
        seg_num = 2
        if L % seg_num != 0:
            raise ValueError(f"Sequence length {L} should be divisible by seg_num {seg_num} for this method.")

        # Step 1: Normalization (by segment)
        x_enc = rearrange(x_enc, 'b (n s) m -> b n s m', s=seg_num)  # Split sequence into segments of size `seg_num`
        means = x_enc.mean(2, keepdim=True).detach()  # Mean along segments
        x_enc = x_enc - means  # Subtract the mean
        stdev = torch.sqrt(torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)  # Standard deviation along segments
        x_enc /= stdev  # Normalize
        x_enc = rearrange(x_enc, 'b n s m -> b (n s) m')  # Flatten segments back

        x_temporal = self.temporal(x_enc)

        # Step 2: Pad to match GPT2's expected input dimension
        padding_size = 768 - x_enc.shape[-1]
        if padding_size > 0:
            x_enc = torch.nn.functional.pad(x_enc, (0, padding_size))

        # Step 3: Construct dynamic prompt based on x_enc
        prompt = self.construct_dynamic_prompt(x_enc)

        # Tokenize the prompt text and create an embedding for it
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt')
        self.prompt_ids = prompt_tokens['input_ids'].squeeze(0).to(x_enc.device)  # Token IDs for the prompt
        self.prompt_embeds = self.gpt2.wte(self.prompt_ids)  # Get embeddings for the prompt

        # Step 4: Prepend the prompt embeddings to the time series data
        prompt_length = self.prompt_embeds.shape[0]
        prompt_embeds = self.prompt_embeds.unsqueeze(0).repeat(B, 1, 1).to(x_enc.device)  # Repeat prompt embeddings for batch size B
        x_enc = torch.cat([prompt_embeds, x_enc], dim=1)  # Concatenate prompt embeddings and time series data

        # Step 5: GPT2 forward pass
        outputs = self.gpt2(inputs_embeds=x_enc).last_hidden_state  # [B, L + prompt_len, D]
        outputs = outputs[:, prompt_length:, :self.d_ff]  # Limit to the first `d_ff` dimensions (output dimension)

        laten_loss, kl_loss, outputs = self.fusion(x_temporal, outputs)

        # Step 6: Apply output layer
        dec_out = self.out_layer(outputs)  # [B, L, c_out]

        # Step 7: De-normalization (reverse the normalization applied in step 1)
        dec_out = rearrange(dec_out, 'b (n s) m -> b n s m', s=seg_num)  # Reshape to segments
        dec_out = dec_out * stdev[:, :, 0, :].unsqueeze(2).repeat(1, 1, seg_num, 1)  # Scale by stdev
        dec_out = dec_out + means[:, :, 0, :].unsqueeze(2).repeat(1, 1, seg_num, 1)  # Add back means
        dec_out = rearrange(dec_out, 'b n s m -> b (n s) m')  # Flatten back to original shape

        # Step 8: Slice the output to match `pred_len`
        dec_out = dec_out[:, -self.pred_len:, :]  # Get the last `pred_len` time steps

        return laten_loss, kl_loss, dec_out.squeeze(1)  # [B, pred_len, c_out]

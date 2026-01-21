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

    def construct_dynamic_prompt(self, x_enc, channel_names=None):
        """
        Construct a structured prompt for EEG seizure prediction based on input data.
        Follows the format specified in the paper/documentation.
        """
        # For private dataset
        if channel_names is None:
            channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 
                            'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'T5', 'T6', 
                            'Pz', 'Cz', 'Fz']
        
        # Compute statistics for each channel
        # x_enc shape: [batch_size, seq_len, num_channels]
        channel_means = x_enc.mean(dim=1)  # Mean across sequence for each channel
        
        # Define anatomical brain region groups
        frontal_indices = [channel_names.index(ch) for ch in ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8'] if ch in channel_names]
        temporal_indices = [channel_names.index(ch) for ch in ['T3', 'T4', 'T5', 'T6'] if ch in channel_names]
        central_indices = [channel_names.index(ch) for ch in ['C3', 'C4'] if ch in channel_names]
        parietal_indices = [channel_names.index(ch) for ch in ['P3', 'P4'] if ch in channel_names]
        occipital_indices = [channel_names.index(ch) for ch in ['O1', 'O2'] if ch in channel_names]
        midline_indices = [channel_names.index(ch) for ch in ['Fz', 'Cz', 'Pz'] if ch in channel_names]
        
        # Calculate group averages
        def calculate_group_average(indices):
            if indices:
                group_vals = channel_means[:, indices]
                return group_vals.mean().item()
            return 0.0
        
        frontal_avg = calculate_group_average(frontal_indices)
        temporal_avg = calculate_group_average(temporal_indices)
        central_avg = calculate_group_average(central_indices)
        parietal_avg = calculate_group_average(parietal_indices)
        occipital_avg = calculate_group_average(occipital_indices)
        midline_avg = calculate_group_average(midline_indices)
        
        # Construct the prompt following the specified format
        prompt = f"""Task: Seizure prediction in EEG data

        Monitored Variables: {', '.join(channel_names)}

        Anatomical Brain Regions Division:
        Frontal group (cognitive control-related activity): Fp1, Fp2, F3, F4, F7, F8
        Temporal group (memory-related processing): T3, T4, T5, T6
        Central group (motor-related activity): C3, C4
        Parietal group (sensory integration): P3, P4
        Occipital group (visual perception): O1, O2
        Midline group (global cognitive state monitoring): Fz, Cz, Pz

        Anomaly Indicators: Sudden signal spikes, disrupted frequency patterns, persistent abnormal patterns and other anomalies.

        Historical Data:

        Feature Group Analysis:
        Frontal Group Avg: {frontal_avg:.4f}
        Temporal Group Avg: {temporal_avg:.4f}
        Central Group Avg: {central_avg:.4f}
        Parietal Group Avg: {parietal_avg:.4f}
        Occipital Group Avg: {occipital_avg:.4f}
        Midline Group Avg: {midline_avg:.4f}

        Anomaly Detection Request:
        Predict the next timestamp's values and identify anomalies. Provide details on:
        - Anomalous variables
        - Expected vs observed behavior"""

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

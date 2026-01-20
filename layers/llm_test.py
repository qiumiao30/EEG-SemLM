import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, LlamaModel
from gat_attention import TemporalGAT
from Embed import PatchEmbedding, PositionalEmbedding
from cross_attention import CrossAttentionFusion
from cross import TemporalSemanticInteraction
from time_predictor import TimePredictor

class EEGPromptGenerator:
    def __init__(self):
        """
        初始化EEG prompt生成器，包含标准21导联名称和相关信息
        """
        # 21导联名称按照国际10-20系统
        self.channels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2', 'A1', 'A2'
        ]
        
        # 定义脑区分组，用于提供更有意义的上下文
        self.brain_regions = {
            'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
            'Central': ['C3', 'Cz', 'C4'],
            'Temporal': ['T3', 'T4', 'T5', 'T6'],
            'Parietal': ['P3', 'Pz', 'P4'],
            'Occipital': ['O1', 'O2'],
            'Auricular': ['A1', 'A2']
        }

    def format_eeg_data(self, timestamp_data):
        """
        格式化单个时间点的EEG数据，不显示通道名称
        """
        return ", ".join([f"{value:.2f}" for value in timestamp_data])

    def get_brain_region_summary(self, timestamp_data):
        """
        生成按脑区分组的数据摘要
        """
        summaries = []
        for region, channels in self.brain_regions.items():
            values = [timestamp_data[self.channels.index(ch)] for ch in channels]
            avg_value = sum(values) / len(values)
            summaries.append(f"{region} Region Avg: {avg_value:.2f}")  # 归一化后的平均值
        return "\n".join(summaries)

    def generate_prompt(self, input_data, window_size=5):
        """
        生成EEG预测的专业prompt
        Args:
            input_data: shape为(batch_size, sequence_length, 21)的EEG数据
            window_size: 用于计算趋势的时间窗口大小
        """
        prompts = []
        
        # 首先，列出所有的EEG通道名，后续数据中将不再重复显示
        channel_names = ", ".join(self.channels)

        for batch_idx in range(input_data.shape[0]):
            # 构建任务描述和上下文
            context = (
                "Task: EEG Signal Prediction\n"
                "Based on the following EEG recordings from 21 channels (International 10-20 system), "
                "predict the next time point's values for all channels.\n\n"
                "Important Context:\n"
                "- Values are normalized between [0, 1] (min-max normalization)\n"
                "- Sampling rate matches clinical EEG standards\n"
                "- Consider both local channel patterns and cross-channel relationships\n\n"
                f"EEG Channels: {channel_names}\n\n"  # 添加通道名称
                "Historical EEG Readings:\n"
            )
            
            # 添加时间序列数据
            time_series_data = []
            for time_step in range(input_data.shape[1]):
                timestamp_data = input_data[batch_idx, time_step]
                
                # 添加详细的时间点数据（只显示数据值，不显示通道名）
                formatted_data = self.format_eeg_data(timestamp_data)
                time_series_data.append(f"\nTimestamp {time_step + 1}:")
                time_series_data.append(formatted_data)
                
                # 每个时间窗口结束时添加脑区摘要
                if (time_step + 1) % window_size == 0 or time_step == input_data.shape[1] - 1:
                    time_series_data.append("\nRegional Analysis:")
                    time_series_data.append(self.get_brain_region_summary(timestamp_data))
            
            # 将时间序列数据添加到上下文中
            context += "\n".join(time_series_data)
            
            # 添加具体预测要求
            request = (
                "\n\nPrediction Request:\n"
                "Based on the above EEG patterns, predict the next timestamp's values "
                "for all 21 channels, considering:\n"
                "1. Individual channel trends and rhythms\n"
                "2. Inter-channel relationships within each brain region\n"
                "3. Overall brain state patterns\n\n"
                "Provide predictions as numerical values for each channel, "
                "maintaining the same channel order as input data. "
                "Format: [value_1, value_2, ..., value_21]"  # 不再显示通道名，只显示数值
            )
            
            prompts.append(context + request)
        
        return prompts

class TemporalPredictionModel(nn.Module):
    def __init__(self, llm_dim, n_feature):
        super(TemporalPredictionModel, self).__init__()
        
        # 定义池化层，假设是全局平均池化
        self.pool = nn.AdaptiveAvgPool1d(1)  # 将 token + series 维度池化为1
        
        # 定义线性投影层，将特征映射到 n_feature
        self.projection_layer = nn.Linear(llm_dim, n_feature)
        
    def forward(self, hidden_states):
        """
        hidden_states: shape (B, token + series, llm_dim + series_dim)
        """
        # 假设我们首先做池化 (将 token + series 维度池化到1)
        pooled = self.pool(hidden_states.permute(0, 2, 1))  # shape (B, llm_dim + series_dim, 1)
        
        # 展平池化后的结果
        pooled = pooled.squeeze(-1)  # shape (B, llm_dim + series_dim)
        
        # 线性投影到预测空间
        predictions = self.projection_layer(pooled)  # shape (B, n_feature)
        
        return predictions

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        llama_model_id = '/home/siat/.cache/modelscope/hub/LLM-Research/Llama-3___2-1B-Instruct'
        
        # Load LLaMA model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_id, padding_side='right')
        self.llama_model = LlamaModel.from_pretrained(llama_model_id)
        
        if not self.tokenizer.pad_token:
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.llama_model.resize_token_embeddings(len(self.tokenizer))
        
        # Freeze LLaMA parameters
        for param in self.llama_model.parameters():
            param.requires_grad = False
        
        # 获取LLaMA模型的hidden_size
        llama_hidden_size = self.llama_model.config.hidden_size

        ##### Prompt生成器
        self.prompt_generator = EEGPromptGenerator()

        ##### Embedding层
        self.pos_embedding = PositionalEmbedding(configs['d_model'])
        self.series_embedding = nn.Linear(configs['enc_in'], configs['d_ff']) 

        ##### GAT_Attention层
        self.gat_attention = TemporalGAT(configs['d_model'], configs['d_model'], 4, 0.2)

        #### Cross-Attention层
        # self.cross_attention = CrossAttentionFusion(hidden_size_llm=2048, hidden_size_time_series=configs['d_model'])
        self.cross_attention = TemporalSemanticInteraction(llm_dim = configs['llm_dim'], dim=configs['d_ff']) #'dual' 或 'co-attention'
        
        # 全局注意力池化层，用于将序列信息聚合
        self.global_attention = nn.Sequential(
            nn.Linear(llama_hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # 预测头
        # self.prediction_head = TemporalPredictionModel(2048, configs['enc_in'])

        self.prediction = TimePredictor(input_dim=configs['d_model'], output_dim=configs['enc_in'])
    
    # @autocast()
    def forward(self, input_data):
        # input_data = input_data.to(torch.float32)

        #### Temporal Encoding
        enc_out = self.pos_embedding(input_data) # [bs, seq_len, d_model]
        x_enc = self.series_embedding(input_data) # [bs, seq_len, d_ff]
        enc_out = enc_out + x_enc
        gat_attention, _, _, _ = self.gat_attention(enc_out)

        # gat_attention = gat_attention.to(torch.float16)

        #### LLM Encoding
        # 生成prompts
        prompts = self.prompt_generator.generate_prompt(input_data)
        # prompts = self.generate_prompt(input_data)
        
        # Tokenize and get LLaMA outputs
        inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(input_data.device)  
        llama_output = self.llama_model(input_ids=inputs["input_ids"], 
                                      attention_mask=inputs["attention_mask"])
        
        # 获取所有hidden states (batch_size, prompt_len, llm_dim)
        hidden_states = llama_output.last_hidden_state
        # hidden_states = hidden_states.to(torch.float16)

        #### Cross-attention fusion
        # weighted_sum = self.cross_attention(hidden_states, gat_attention)
        _, _, output = self.cross_attention(hidden_states, gat_attention)
        
        #### 通过预测头生成最终预测
        # predictions = self.prediction_head(output)  # (batch_size, num_features)
        predictions = self.prediction(output)  # (batch_size, num_features)
        
        return predictions
    

configs = {
    'task_name': 'long_term_forecast',
    'pred_len': 1, 
    'seq_len': 10,
    'enc_in': 21,
    'd_ff': 512,
    'llm_dim': 2048, # qwen: 896
    'llm_layers': 1,
    'dropout': 0.1,
    'd_model': 512,
    'n_heads': 4,
    'model_id': '/home/siat/.cache/modelscope/hub/Qwen/Qwen2___5-0___5B-Instruct', 
    # 'model_id': '/home/siat/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct',
    # 'model_id': '/home/siat/.cache/modelscope/hub/LLM-Research/Llama-3___2-1B-Instruct',
    'num_tokens': 1000,
    # 'model_id': 'Qwen', # Qwen, Llama
    }   


model = Model(configs).cuda()
model = torch.compile(model).to(torch.float16)

input_data = torch.randn(2, 10, 21).to(torch.float16).cuda()
output = model(input_data)

print(output)
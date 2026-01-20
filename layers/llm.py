import torch
from math import sqrt
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, AutoTokenizer, BitsAndBytesConfig, LlamaTokenizer, GPT2Config, GOT2Model, GPT2Tokenizer
from layers.Embed import PatchEmbedding, PositionalEmbedding
from layers.StandardNorm import Normalize
# from layers.generate_prompt import PromptGenerator
from layers.FreMlp import MlpModel
from layers.prompt import PromptGenerator
from layers.fusion import MutualInfoModel
from layers.patchformer import ForcastConvTransformer

import transformers
transformers.logging.set_verbosity_error() # suppress warnings


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, n_vars)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
    
    
class MultiHeadSelfAttentionCustom(nn.Module):
    def __init__(self, patch_dim, d_ff, num_heads):
        super(MultiHeadSelfAttentionCustom, self).__init__()
        assert patch_dim % num_heads == 0, "patch_dim 必须能被 num_heads 整除"
        
        self.num_heads = num_heads
        self.head_dim = patch_dim // num_heads  # 每个头的维度
        self.scale = self.head_dim ** -0.5      # 缩放因子

        # 定义用于生成 Q、K、V 的线性层
        self.q_linear = nn.Linear(patch_dim, patch_dim)
        self.k_linear = nn.Linear(patch_dim, patch_dim)
        self.v_linear = nn.Linear(patch_dim, patch_dim)

        # 最终的线性层，用于拼接多个头后的映射
        self.fc_out = nn.Linear(patch_dim, d_ff)

    def forward(self, x):
        batch_size, patch_num, patch_dim = x.shape  # x 形状: (bs, seq_num, dim)
        
        # 1. 通过线性层生成 Q, K, V
        Q = self.q_linear(x)  # (bs , c)
        K = self.k_linear(x)  # (bs , seq_num, dim)
        V = self.v_linear(x)  # (bs , seq_num, dim)
        
        # 2. 将 Q, K, V 拆分成多个头
        Q = Q.view(batch_size, patch_num, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, seq_num, dim)
        K = K.view(batch_size, patch_num, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, seq_num, dim)
        V = V.view(batch_size, patch_num, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, num_heads, seq_num, dim)
        
        # 3. 计算 Q 和 K 的点积 (scaled dot-product attention)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (bs, num_heads, seq_num, seq_num)
        attn_weights = F.softmax(attn_weights, dim=-1)  # 在最后一个维度上做 softmax
        
        # 4. 用注意力权重加权 V
        attn_output = torch.matmul(attn_weights, V)  # (bs, num_heads, patch_num, head_dim)
        
        # 5. 拼接多个头的输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, patch_num, patch_dim)  # (bs, seq_num, dim)
        
        # 6. 通过最终线性层映射到 d_ff 维度
        output = self.fc_out(attn_output)  # (bs * n_vars, patch_num, d_ff)
        return output, attn_weights

class Model(nn.Module):
    def __init__(self, configs, patch_len=7, stride=3):
        super(Model, self).__init__()
        self.task_name = configs['task_name']
        self.pred_len = configs['pred_len']
        self.seq_len = configs['seq_len']
        self.d_ff = configs['d_ff']
        self.top_k = 5
        self.d_llm = configs['llm_dim']
        self.patch_len = patch_len
        self.stride = stride
        # model_id = '/home/siat/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct'
        model_id = '/home/siat/.cache/modelscope/hub/LLM-Research/Llama-3___2-1B-Instruct'
        # model_id = '/home/siat/.cache/modelscope/hub/LLM-Research/Llama-3___2-1B'
        # model_id = '/home/siat/.cache/modelscope/hub/Qwen/Qwen2___5-0___5B-Instruct'
        self.llama_config = LlamaConfig.from_pretrained(model_id)
        self.llama_config.num_hidden_layers = configs['llm_layers']
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True

        if configs['llm_model'] == 'llama':
            self.llm_model = LlamaModel.from_pretrained(model_id, config=self.llama_config)
            self.llama_config.num_hidden_layers = configs['llm_layers']
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True

            try:
                self.llm_model = LlamaModel.from_pretrained(model_id, 
                                                            trust_remote_code=True,
                                                            local_files_only=True,
                                                            config=self.llama_config
                                                            )
            except EnvironmentError:
                print("local model files not found, downloading from remote server")
                self.llm_model = LlamaModel.from_pretrained('huggllama/llama-7b', 
                                                            trust_remote_code=True,
                                                            local_files_only=False,
                                                            config=self.llama_config
                                                            )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(model_id,
                                                                trust_remote_code=True,
                                                                local_files_only=True
                                                                )
            except EnvironmentError:
                print("local tokenizer files not found, downloading from remote server")
                self.tokenizer = LlamaTokenizer.from_pretrained('huggllama/llama-7b',
                                                                trust_remote_code=True,
                                                                local_files_only=False
                                                                )
                
        elif configs['llm_model'] == 'gpt2':
            self.gpt2_confg = GPT2Config.from_pretrained('opemai-community/gpt2')
            self.gpt2_confg.num_hidden_layers = configs['llm_layers']
            self.gpt2_confg.output_attentions = True
            self.gpt2_confg.output_hidden_states = True

            try:
                self.llm_model = GOT2Model.from_pretrained('opemai-community/gpt2', 
                                                            trust_remote_code=True,
                                                            local_files_only=True,
                                                            config=self.gpt2_confg
                                                            )
            except EnvironmentError:
                print("local model files not found, downloading from remote server")
                self.llm_model = GOT2Model.from_pretrained('opemai-community/gpt2', 
                                                            trust_remote_code=True,
                                                            local_files_only=False,
                                                            config=self.gpt2_confg
                                                            )
            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained('opemai-community/gpt2',
                                                                trust_remote_code=True,
                                                                local_files_only=True
                                                                )
            except EnvironmentError:
                print("local tokenizer files not found, downloading from remote server")
                self.tokenizer = GPT2Tokenizer.from_pretrained('opemai-community/gpt2',
                                                                trust_remote_code=True,
                                                                local_files_only=False
                                                                )
        else:
            raise Exception("llm_model not found")
        
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        self.generator = PromptGenerator()

        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(configs['dropout'])

        # Normalization layer
        self.layer_norm1 = nn.LayerNorm(configs['d_ff'])
        self.layer_norm2 = nn.LayerNorm(configs['llm_dim'])

        self.pos_embedding = PositionalEmbedding(configs['d_model'])

        self.patch_embedding = PatchEmbedding(
            configs['d_model'], self.patch_len, self.stride, configs['dropout'])

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(configs['d_model'], configs['n_heads'], self.d_ff, self.d_llm)

        self.fre_mlp = MlpModel(pred_len=1, enc_in=configs['enc_in'], seq_len=configs['seq_len'], channel_independence='1', embed=configs['d_model'])
        self.attention_layer = MultiHeadSelfAttentionCustom(configs['d_model'], self.d_ff, configs['n_heads'])
        self.temporal = ForcastConvTransformer(configs['enc_in'], configs['d_model'], configs['n_heads'], configs['n_layers'], configs['seq_len'], 
                                               kernel_size=3, mask_next=True, mask_diag=False, dropout_proba=0.2)

        self.patch_regramming_layer = nn.Linear(512, 2048)

        self.head_nf = configs['enc_in'] * configs['seq_len']

        self.series_embedding = nn.Linear(configs['enc_in'], self.d_ff)  

        self.llmdim2d_ff = nn.Linear(self.d_llm, self.d_ff)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs['enc_in'], self.head_nf, 
                                                 head_dropout=configs['dropout'])
        else:
            raise NotImplementedError
        
        self.dff2feature = nn.Linear(self.d_ff * 2, configs['enc_in'])

        self.normalize_layers = Normalize(configs['enc_in'], affine=False)


        self.fusion = MutualInfoModel(configs['d_model'], configs['d_model'], seq_len=configs['seq_len'], latent_size=configs['d_model'])

    def forward(self, x_enc):
        x_enc = self.normalize_layers(x_enc, 'norm')
        # x_enc = self.MotionBlocks(x_enc)
        x_enc = x_enc.to(torch.float32)
        B, T, N = x_enc.size()

        prompt = self.generator.generate_prompt(x_enc)

        x_enc = x_enc.to(torch.bfloat16)

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids # [bs, prompt_len]
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # [bs, prompt_len, llm_dim]

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)  # [num_tokens, llm_dim]
        # source_embeddings = source_embeddings.unsqueeze(0).expand(B, -1, -1)  # [bs, num_tokens, llm_dim]

        enc_out = self.reprogramming_layer(x_enc, source_embeddings, source_embeddings)  # [bs, seq_len, llm_dim]
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1) # [bs, prompt_len+seq_len, llm_dim]
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state # [bs, prompt_len+seq_len, llm_dim]
        dec_out = dec_out[:, :, :self.d_ff] #


        enc_out_attention = self.temporal(x_enc) # [bs, seq_len, d_ff]

        dec_out = self.llmdim2d_ff(dec_out) # [bs, prompt_len+patch_num, d_ff]

        ###  prompt
        dec_out = dec_out[:, -T:, :] #
        
        
        latent_loss, kl_loss, z_combined = self.fusion(enc_out_attention, dec_out)
        
        # dec_out = dec_out.to(torch.bfloat16)

        z_combined = self.dff2feature(z_combined) # [bs, n_vars, enc_in]
        z_combined = self.normalize_layers(z_combined, 'denorm')

        dec_out = self.output_projection(z_combined) # [bs, n_vars, pred_len]
        
        dec_out = dec_out.to(torch.bfloat16)

        return latent_loss, kl_loss, dec_out
    

llm_model = Model(configs={'task_name': 'long_term_forecast', 'pred_len': 1, 'seq_len': 10, 'd_ff': 512, 'llm_dim': 1024, 'dropout': 0.1, 'n_heads': 8, 'llm_layers': 6, 'enc_in': 256, 'n_layers': 2, 'llm_model': 'gpt2'})

input = torch.randn(32, 10, 256)

configs = {'task_name': 'long_term_forecast', 'pred_len': 1, 
           'seq_len': 10, 'd_ff': 512, 'llm_dim': 1024, 'dropout': 0.1, 
           'n_heads': 8, 'llm_layers': 6, 'enc_in': 256, 'n_layers': 2, 'llm_model': 'gpt2'}

print(llm_model(input))
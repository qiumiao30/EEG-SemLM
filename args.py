import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def get_parser():
    parser = argparse.ArgumentParser()

    # -- Data params ---output/EEG/27112024_101433/output/EEG/27112024_101433/
    parser.add_argument("--dataset", type=str.upper, default="EEG2")
    parser.add_argument("--group", type=str, default="1-1", help="Required for SMD dataset. <group_index>-<index>")
    parser.add_argument("--index", type=str, default="1", help="Required for omi dataset. omi-<index>")
    parser.add_argument("--lookback", type=int, default=200, help="Windows size")
    parser.add_argument("--normalize", type=str2bool, default=True)
    parser.add_argument("--spec_res", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=100)

    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--training_list', type=str, default='execute_list/train_all.csv', help='list of the training tasks')
    parser.add_argument('--inference_list', type=str, default='execute_list/inference_all.csv', help='list of the inference tasks')
    parser.add_argument('--eval_model_path', type=str, default='', help='pretrain model path for evaluation')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    # parser.add_argument('--seed', type=int, default=2036, help='random seed')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--label_len', type=int, default=0, help='label length')

    # model
    parser.add_argument('--lm_pretrain_model', type=str, default='gpt2-small', help='pretrain model name')
    parser.add_argument('--lm_ft_type', type=str, default='full', help='fine-tuning type, options:[freeze: all parameters freeze, fpt: only tune positional embeddings and layernorms, full: full parameters tuning]')
    parser.add_argument('--instruct_path', type=str, default='data_configs/instruct.json', help='instruction list')
    parser.add_argument('--zero_shot_instruct', type=str, default='', help='zero shot instruction')
    parser.add_argument('--mask_rate', type=float, default=0.5, help='masking rate')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--max_token_num', type=int, default=0, help='maximum token number')
    parser.add_argument('--max_backcast_len', type=int, default=96, help='maximum backcast sequence length')
    parser.add_argument('--max_forecast_len', type=int, default=720, help='maximum forecast sequence length')
    parser.add_argument('--lm_layer_num', type=int, default=6, help='language model layer number')
    parser.add_argument('--dec_trans_layer_num', type=int, default=2, help='decoder transformer layer number')
    parser.add_argument('--ts_embed_dropout', type=float, default=0.3, help='time series embedding dropout')
    parser.add_argument('--dec_head_dropout', type=float, default=0.1, help='decoder head dropout')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--patience', type=int, default=10, help='early stop patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--clip', type=int, default=5, help='gradient clipping')

    # -- Model params ---
    # model select
    parser.add_argument("--model", type=str, default="llm", help="Model to use")
    # 1D conv layer
    parser.add_argument("--kernel_size", type=int, default=5)
    # GAT layers
    parser.add_argument("--use_gatv2", type=str2bool, default=True)
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--time_gat_embed_dim", type=int, default=None)
    # GRU layer
    parser.add_argument("--gru_n_layers", type=int, default=2)
    parser.add_argument("--gru_hid_dim", type=int, default=64)
    # parser.add_argument("--is_training", type=float, default=1)

    parser.add_argument("--hid_dim", type=int, default=64)
    # Forecasting Model
    parser.add_argument("--fc_n_layers", type=int, default=1)
    parser.add_argument("--fc_hid_dim", type=int, default=64)
    # Reconstruction Model
    parser.add_argument("--recon_n_layers", type=int, default=1)
    parser.add_argument("--recon_hid_dim", type=int, default=150)
    # Other
    parser.add_argument("--alpha", type=float, default=0.2)

    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--header", type=int, default=8)

    # --- Train params ---
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--val_split", type=float, default=0.9)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--log_tensorboard", type=str2bool, default=True)

    # --- Predictor params ---
    parser.add_argument("--scale_scores", type=str2bool, default=True)
    parser.add_argument("--use_mov_av", type=str2bool, default=True)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--level", type=float, default=None)
    parser.add_argument("--q", type=float, default=None)
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)
    parser.add_argument("--adjust_score", type=str2bool, default=True)
    parser.add_argument("--k", type=int, default=0)

    # --- Other ---
    parser.add_argument("--comment", type=str, default="")

    return parser

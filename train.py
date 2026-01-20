import json
from datetime import datetime
import torch.nn as nn
import random
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import os

from args import get_parser
from utils import *

from prediction import Predictor
from training import Trainer
from model.llmanomaly import Model


torch.manual_seed(0)
if __name__ == "__main__":

    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()

    random.seed(args.seed)              
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)         
    torch.cuda.manual_seed(args.seed)   
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False      
    torch.backends.cudnn.deterministic = False   
    os.environ['PYTHONHASHSEED'] = str(args.seed) 

    dataset = args.dataset
    window_size = args.lookback
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index = args.group[0]
    index = args.group[2:]
    #index = args.index
    depth = args.depth
    header = args.header
    kernel_size = args.kernel_size
    dropout = args.dropout
    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    args_summary = str(args.__dict__)
    print(args_summary)

    
    if dataset == "EEG":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    elif dataset == "EEG2":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    x_train = x_train.values

    x_test = x_test.values
    y_test = y_test

    # 转换为tensor
    if args.val_split is not None:
        dataset_size = len(x_train)
        split = int(np.floor(args.val_split * dataset_size))
        train_data, val_data = x_train[:split], x_train[split:]
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")

    # x_train = torch.from_numpy(train_data).float()

    if args.model == "llm":

        x_train = torch.tensor(train_data)
        x_val = torch.tensor(val_data)
        x_test = torch.tensor(x_test)
    else:
        x_train = torch.tensor(train_data, dtype=torch.float32)
        x_val = torch.tensor(val_data, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)


    # x_val = torch.from_numpy(val_data).float()
    # x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1] # 特征

    target_dims = get_target_dims(dataset)
    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims, stride=args.stride)
    val_dataset = SlidingWindowDataset(x_val, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)
    

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, batch_size, shuffle_dataset, test_dataset=test_dataset
    )

    # Example usage with random data

    pred_len = 1
    seq_len = window_size
    enc_in = n_features
    batch_size = batch_size
    
    
    configs = {
        'd_ff': 128,
        'seq_len': window_size,
        'pred_len': 1,
        'gpt_layers': 1,
        'c_out': 1,
        'enc_in': n_features,
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 2,
        'train_gpt2': False
    }

    
    if args.model == "llm":
        model = Model(configs)
    else:
        pass

    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    forecast_criterion = nn.MSELoss()

    trainer = Trainer(args,
        dataset,
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )
#############################################
    trainer.fit(train_loader, val_loader)

    plot_losses(trainer.losses, save_path=save_path, plot=False)

###############################################

    # Some suggestions for POT args
    level_q_dict = {
        "EEG": (0.95, 0.05),
        "EEG2": (0.95, 0.05),
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"EEG":1, "EEG2":1}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]

    trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
        "k": args.k,
        "adjust_score": args.adjust_score,
        "batch_size": args.bs,
        "model": args.model,
    }
    best_model = trainer.model
    predictor = Predictor(
        best_model,
        window_size,
        n_features,
        prediction_args,
    )

    label = y_test[window_size:] if y_test is not None else None

    predictor.predict_anomalies(x_train, x_test, label)

    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)


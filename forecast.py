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

def data_provider(dataset, normalize=True):

    (x_train, _), (x_test, y_test), series_cols = get_data(dataset, normalize=normalize)

    x_train = x_train.values[:1000,:]
    x_test = x_test.values[:1000,:]
    y_test = y_test[:1000]

    # 转换为tensor
    if args.val_split is not None:
        dataset_size = len(x_train)
        split = int(np.floor(args.val_split * dataset_size))
        train_data, val_data = x_train[:split], x_train[split:]
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Val data shape: {val_data.shape}")

    # x_train = torch.from_numpy(train_data).float()
    x_train = torch.tensor(train_data, dtype=torch.float32).to(torch.bfloat16)
    x_val = torch.tensor(val_data, dtype=torch.float32).to(torch.bfloat16)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(torch.bfloat16)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims, stride=args.stride)
    val_dataset = SlidingWindowDataset(x_val, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, batch_size, shuffle_dataset, test_dataset=test_dataset
    )

    return train_loader, val_loader, test_loader

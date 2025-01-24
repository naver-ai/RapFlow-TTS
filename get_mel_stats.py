# reference: https://github.com/shivammehta25/Matcha-TTS

r"""
The file creates a pickle file where the values needed for loading of dataset is stored and the model can load it
when needed.

Parameters from hparam.py will be used
"""
import argparse
import json
import os
import sys
import torch
from tqdm.auto import tqdm
from src.utils import *
from src.dataset import *



def compute_data_statistics(data_loader: torch.utils.data.DataLoader, out_channels: int):
    """Generate data mean and standard deviation helpful in data normalisation

    Args:
        data_loader (torch.utils.data.Dataloader): _description_
        out_channels (int): mel spectrogram channels
    """
    total_mel_sum = 0
    total_mel_sq_sum = 0
    total_mel_len = 0

    for batch in tqdm(data_loader, leave=False):
        mels = batch["y"]
        mel_lengths = batch["y_lengths"]

        total_mel_len += torch.sum(mel_lengths)
        total_mel_sum += torch.sum(mels)
        total_mel_sq_sum += torch.sum(torch.pow(mels, 2))

    data_mean = total_mel_sum / (total_mel_len * out_channels)
    data_std = torch.sqrt((total_mel_sq_sum / (total_mel_len * out_channels)) - torch.pow(data_mean, 2))

    return {"mel_mean": data_mean.item(), "mel_std": data_std.item()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str,   default='./config/LJSpeech/base_stage1.yaml', help='synthesize path')
    parser.add_argument('--num_worker', type=int, default=0, help='Num workers')
    parser.add_argument('--seed', type=int, default=100, help='seed number')
    args = parser.parse_args()
    cfg  = Config(args.config)

    args   = get_params(args)
    for key in args:
        cfg[key] = args[key]

    print(cfg)
    
    batch_collate = TextMelSpeakerBatchCollate(cfg.model.n_spks)
    train_dataset = TextMelSpeakerDataset(cfg.path.train_path, cfg)
    train_loader  = DataLoader(dataset=train_dataset, batch_size=int(cfg.train.batch_size), collate_fn=batch_collate, 
                               num_workers=cfg.num_worker, shuffle=False)

    params = compute_data_statistics(train_loader, cfg.model.n_feats)
    print(params)
    json.dump(
        params,
        open('stats.json', "w"),
    )


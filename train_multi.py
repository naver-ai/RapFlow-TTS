"""
RapFlow-TTS
Copyright (c) 2025-present NAVER Cloud Corp.
Apache-2.0
"""

import os
import sys

# os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
# os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import warnings
warnings.filterwarnings('ignore')

import json
import yaml
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from copy import deepcopy
from time import time
import logging

from scipy.io.wavfile import write

import shutil
from text.symbols import symbols
from model.utils import fix_len_compatibility
from src.utils import *
from src.dataset import *
from argument import *
from model import RapFlowTTS
from model.utils import * 
from collections import OrderedDict
from torch.cuda.amp import autocast, GradScaler 
from tqdm import tqdm
from hifigan.denoiser import Denoiser



@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params   = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir=None):
    """Create a logger that writes to both stdout and a log file if specified."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # StreamHandler to output to stdout
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler to write to file if logging_dir is provided
    if logging_dir is not None and dist.get_rank() == 0:  # Only the main process writes to the log file
        os.makedirs(logging_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(logging_dir, "log.txt"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def main(args, cfg):
    
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group(backend='nccl')
    torch.backends.cudnn.benchmark = True  
    assert cfg.train.batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank        = dist.get_rank()
    device      = rank % torch.cuda.device_count()
    seed        = args.seed * dist.get_world_size() + rank
    
    seed_init(seed=seed)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    torch.cuda.set_device(device)
    
    if rank == 0:
        cfg  = set_experiment(args, cfg, phase='train')
        shutil.copyfile(args.config, os.path.join(cfg.checkpoint, 'base.yaml'))
        
        logger = create_logger(cfg.checkpoint)
        logger.info(f"Experiment directory created at {cfg.checkpoint}")
        logger.info("==== Paramter Settings ====")
        logger.info(cfg)
    else:
        logger = create_logger(None)
        args   = get_params(args)
        for key in args:
            cfg[key] = args[key]

    model = RapFlowTTS(cfg.model)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  ## convert to sync batchnorm
    ema   = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    if cfg.train.encoder_freeze:
        model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    else:
        model = DDP(model.to(device), device_ids=[rank])
        
    param_size     = count_parameters(model)
    cfg.param_size = np.round(param_size/1000000,2)
    logger.info(f"Model Parameters: {cfg.param_size}M")
    
    vis_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    logger.info(f"Model training on Device number: {vis_devices}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train.lr))
    scaler    = GradScaler(enabled=cfg.train.amp)
    
    batch_collate = TextMelSpeakerBatchCollate(cfg.model.n_spks)
    train_dataset = TextMelSpeakerDataset(cfg.path.train_path, cfg)
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=cfg.seed)
    train_loader  = DataLoader(dataset=train_dataset, batch_size=int(cfg.train.batch_size // dist.get_world_size()), collate_fn=batch_collate, 
                               sampler=train_sampler, num_workers=cfg.num_worker, shuffle=False, pin_memory=True, drop_last=True)

    run = None
    if rank == 0:
        val_dataset  = TextMelSpeakerDataset(cfg.path.val_path, cfg)
        val_loader   = DataLoader(dataset=val_dataset, batch_size=int(cfg.train.batch_size), collate_fn=batch_collate, 
                                  num_workers=cfg.num_worker, shuffle=False, pin_memory=True, drop_last=True)
        vocoder      = get_vocoder(cfg, f'cuda:{device}')
        denoiser     = Denoiser(vocoder, mode='zeros')  ## get denoiser
        if cfg.logging:
            logger.info('---logging start---')
            run = neptune_load(get_cfg_params(cfg))

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # checkpoint
    epoch_start = 1
    if cfg.train.stage == 2:
        ### full training stage 2
        ckpt_name  = os.path.basename(cfg.train.prev_stage_ckpt)
        checkpoint = torch.load(f'{cfg.train.prev_stage_ckpt}', map_location=f'cuda:{device}')
        model.module.load_state_dict(checkpoint['state_dict'], strict=False)
        ema.load_state_dict(checkpoint['ema'], strict=False)
        
        # encoder freeze
        if cfg.train.encoder_freeze:
            for param in model.module.encoder.parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(model.module.decoder.parameters(), lr=float(cfg.train.lr))
            logger.info(f'---Encoder Freeze---')
        logger.info(f'---checkpoint path {cfg.train.prev_stage_ckpt} ---')
        logger.info(f'---load previous weigths {ckpt_name} and new optimizer for training stage 2---')
            
    if cfg.resume is not None:
        if rank != 0:
            ex_name        = os.path.basename(os.getcwd())
            cfg.ex_name    = f'{ex_name}-{cfg.resume}'
            cfg.checkpoint = os.path.join(cfg.checkpoint, cfg.ex_name)
            
        checkpoint = torch.load(f'{cfg.checkpoint}/model-last.pth', map_location=f'cuda:{device}')
        model.module.load_state_dict(checkpoint['state_dict'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer']) 
        epoch_start = checkpoint['epoch'] + 1
        logger.info('---load previous weigths and optimizer for resume training---')
        logger.info(f'---the resume epoch is {epoch_start}---')
        
        
    # Variables for monitoring/logging purposes:
    best_train_loss = 1000000
    cur_step        = 0

    logger.info(f"Training for {cfg.train.epoch} epochs...")
    for epoch in range(epoch_start, cfg.train.epoch+1):
        train_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        
        total_dur_loss   = 0
        total_prior_loss = 0
        total_diff_loss  = 0
        for i, batch in enumerate(tqdm(train_loader)):

            x, x_lengths     = batch['x'].to(device), batch['x_lengths'].to(device)
            y, y_lengths     = batch['y'].to(device), batch['y_lengths'].to(device)
            spk              = batch['spks'].to(device) if batch["spks"] is not None else None

            with autocast(enabled=cfg.train.amp):
                dur_loss, prior_loss, diff_loss, x1, x1_pred = model(x, x_lengths, y, y_lengths, spks=spk, out_size=cfg.train.out_size, cur_epoch=epoch)
                if cfg.train.encoder_freeze:
                    dur_loss, prior_loss = torch.tensor(0).to(diff_loss.device), torch.tensor(0).to(diff_loss.device)                  
                loss = sum([dur_loss, prior_loss, diff_loss])
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if cfg.train.encoder_freeze:
                    torch.nn.utils.clip_grad_norm_(model.module.decoder.parameters(), max_norm=cfg.train.max_grad)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.max_grad)
                scaler.step(optimizer)
                scaler.update()
                if rank == 0:
                    update_ema(ema, model.module)
                
                #############################

            cur_step         += 1
            total_dur_loss   += dur_loss.item()
            total_diff_loss  += diff_loss.item()
            total_prior_loss += prior_loss.item() 

        
        total_dur_loss   = total_dur_loss / (i+1)
        total_diff_loss  = total_diff_loss / (i+1)
        total_prior_loss = total_prior_loss / (i+1)
        train_loss       = (total_dur_loss + total_diff_loss + total_prior_loss) / 3
        
        if rank==0:
            logger.info(f"(Step: {cur_step:07d}) | train Loss: {train_loss:.4f} | dur Loss: {total_dur_loss:.4f} | diff Loss: {total_diff_loss:.4f} | prior Loss: {total_prior_loss:.4f}")
            if cfg.logging:
                run['cur epoch'].append(epoch)
                run['train/total loss'].append(train_loss)
                run['train/dur loss'].append(total_dur_loss)
                run['train/diff loss'].append(total_diff_loss)
                run['train/prior loss'].append(total_prior_loss)
                
            validate(model, val_loader, run, logger, cur_step, epoch, cfg, cfg.device)
        
            checkpoint = {
                        'epoch': epoch,
                        'state_dict': model.module.state_dict(),
                        'ema':        ema.state_dict(),
                        'optimizer':  optimizer.state_dict()} 


            torch.save(checkpoint, f'{cfg.checkpoint}/model-last.pth')
            if epoch % cfg.train.save_epoch == 0:
                torch.save(checkpoint, f'{cfg.checkpoint}/model-train-{epoch}.pth')                    
            if epoch % cfg.train.syn_every == 0:    
                synthesize(model.module, vocoder, denoiser, val_dataset, logger, cfg, device, sample_size=10)
        dist.barrier()
    model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")
    if cfg.logging and rank==0:
        run.stop()
    cleanup()
    
def validate(model, data_loader, run, logger, cur_step, cur_epoch, cfg, device):
    
    total_dur_loss   = 0
    total_prior_loss = 0
    total_diff_loss  = 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):

            x, x_lengths     = batch['x'].to(device), batch['x_lengths'].to(device)
            y, y_lengths     = batch['y'].to(device), batch['y_lengths'].to(device)
            spk              = batch['spks'].to(device) if batch["spks"] is not None else None

            with autocast(enabled=cfg.train.amp):
                dur_loss, prior_loss, diff_loss, x1, x1_pred = model(x, x_lengths, y, y_lengths, spks=spk, out_size=cfg.train.out_size, cur_epoch=cur_epoch)
                if cfg.train.encoder_freeze:
                    dur_loss, prior_loss = torch.tensor(0).to(diff_loss.device), torch.tensor(0).to(diff_loss.device) 
        
            total_dur_loss   += dur_loss.item()
            total_diff_loss  += diff_loss.item()
            total_prior_loss += prior_loss.item() 
        
        total_dur_loss   = total_dur_loss / (i+1)
        total_diff_loss  = total_diff_loss / (i+1)
        total_prior_loss = total_prior_loss / (i+1)
        valid_loss       = (total_dur_loss + total_diff_loss + total_prior_loss) / 3
        
        logger.info(f"(Step: {cur_step:07d}) | valid Loss: {valid_loss:.4f} | dur Loss: {total_dur_loss:.4f} | diff Loss: {total_diff_loss:.4f} prior Loss: {total_prior_loss:.4f}")
        if cfg.logging:
                run['valid/total loss'].append(valid_loss)
                run['valid/dur loss'].append(total_dur_loss)
                run['valid/diff loss'].append(total_diff_loss)
                run['valid/prior loss'].append(total_prior_loss)
    model.train()
        
def synthesize(model, vocoder, denoiser, dataset, logger, cfg, device, sample_size=10):

    # we synthesize samples with current model
    syn_path = f'{cfg.sample_path}/syn/'
    img_path = f'{cfg.sample_path}/img/'
    MakeDir(syn_path)
    MakeDir(img_path)

    data_size   = len(dataset)
    filelist    = dataset.filelist
    sampled_idx = random.sample(range(data_size), k=sample_size)

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(sampled_idx):
            one_batch = dataset.get_datapoint(filelist[idx])
            x         = one_batch["x"].unsqueeze(0).to(device)
            x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=device)            
            spks      = torch.tensor([one_batch["spk"]], dtype=torch.long, device=device) if model.n_spks > 1 else None
            
            output    = model.synthesise(x[:, :x_lengths], x_lengths, n_timesteps=cfg.n_timesteps, spks=spks, temperature=cfg.temperature)
            y_enc, y_dec, attn, mel = output["encoder_outputs"], output["decoder_outputs"], output["attn"], output["mel"]
            
            spks      = spks.item() if one_batch["spk"] is not None else '0'
            spks      = str(spks)
            basename = str(spks)+'_'+str(i)+'.wav'
            audio = vocoder(mel).clamp(-1, 1)
            audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
            syn_save_path  = f'{syn_path}/{basename}'
            sf.write(syn_save_path, audio, 22050, 'PCM_24')

            ## save_image
            y = one_batch["y"]
            save_plot(y, os.path.join(img_path, f'{spks}_{str(i)}_mel.png'))
            save_plot(y_enc.squeeze().cpu(), os.path.join(img_path, f'{spks}_{str(i)}_y_enc.png'))
            save_plot(y_dec.squeeze().cpu(), os.path.join(img_path, f'{spks}_{str(i)}_y_dec.png'))
            save_plot(attn.squeeze().cpu(), os.path.join(img_path, f'{spks}_{str(i)}_attn.png'))

    model.train()

    
if __name__ == "__main__":
    
    args = get_config()
    cfg  = Config(args.config)

    if cfg.train.out_size:
        cfg.train.out_size = fix_len_compatibility(cfg.train.fix_len * cfg.preprocess.sample_rate // cfg.preprocess.hop_length) 
    else:
        cfg.train.out_size = None

    main(args, cfg)

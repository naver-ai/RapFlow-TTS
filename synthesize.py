"""
RapFlow-TTS
Copyright (c) 2025-present NAVER Cloud Corp.
Apache-2.0
"""
    
import os
import warnings
warnings.filterwarnings('ignore')

import argparse
from src.utils import *
from text import text_to_sequence, cleaned_text_to_sequence, sequence_to_text
from model import RapFlowTTS
from scipy.io.wavfile import write
from hifigan.denoiser import Denoiser

global MAX_VALUE  
MAX_VALUE= 32768.0
           
def main(cfg):
    
    MakeDir(cfg.wav_path)
    
    seed_init(seed=cfg.seed)   
    model = RapFlowTTS(cfg.model).to(cfg.device)
    ckpt  = torch.load(os.path.join(cfg.weight_path, f'{cfg.weight_name}.pth'), map_location=cfg.device)
    if cfg.test.ema:
        model.load_state_dict(ckpt['ema'])
    else:
        model.load_state_dict(ckpt['state_dict'], strict=False)

    vocoder   = get_vocoder(cfg, cfg.device)
    denoiser  = Denoiser(vocoder, mode='zeros')  ## get denoiser

    text    = cfg.input_text
    print(text)
    
    x = text_to_sequence(text, cfg.preprocess.cleaner)[0]
    if cfg.model.add_blank:
        x = torch.tensor(intersperse(x, 0), dtype=torch.long, device=cfg.device)[None]
    else:
        x = torch.tensor(x, dtype=torch.long, device=cfg.device)[None]
    x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=cfg.device)
    spk       = torch.tensor([int(cfg.spk_id)],dtype=torch.long, device=cfg.device) if model.n_spks > 1 else None

    model.eval()
    with torch.no_grad():
        output = model.synthesise(x, x_lengths, n_timesteps=cfg.n_timesteps, temperature=cfg.temperature, spks=spk, length_scale=cfg.length_scale)
        output['waveform'] = vocoder(output['mel']).clamp(-1, 1)
        output['waveform'] = denoiser(output['waveform'].squeeze(0), strength=0.00025).cpu().squeeze().numpy()
    

    basename    = str(spk.cpu().numpy()[0]) if spk is not None else '0'
    output_name = basename + '_syn.wav'
    output_path = os.path.join(cfg.wav_path, output_name)

    write(output_path, 22050, output['waveform'])
    print('Done. Check out `out` folder for samples.')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',        type=str,   default='cuda:0', help='Device')
    parser.add_argument('--wav_path',      type=str,   default='./syn_samples', help='synthesize path')
    parser.add_argument('--spk_id',        type=int,   default=0, help='Pre-defined speaker ID')
    parser.add_argument('--weight_path',   type=str,   default='./checkpoints/RapFlow-TTS-LJS-Stage3-Improved', help='pre-trained weight path')
    parser.add_argument('--model_name',    type=str,   default='RapFlowTTS', help='pre-trained weight path')
    parser.add_argument('--seed',          type=int,   default=100, help='seed number')
    parser.add_argument('--n_timesteps',   type=int,   default=2, help='Time step')
    parser.add_argument('--temperature',   type=float, default=0.667, help='Temperature')
    parser.add_argument('--length_scale',  type=float, default=1.0, help='length scale')
    parser.add_argument('--weight_name',   type=str,   default='model-last', help='length scale')
    parser.add_argument('--input_text',   type=str,   default='Rap-flow TTS is a TTS model using improved consistency flow matching, and it can synthesize high-quality speech with fewer steps.', help='input text')
    args = parser.parse_args()
        
    cfg_path = os.path.join(args.weight_path, 'base.yaml')
    cfg  = Config(cfg_path)
    args = get_params(args)
    for key in args:
        cfg[key] = args[key]    


    print(cfg)
    main(cfg)


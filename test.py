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
from text.symbols import symbols
from model import RapFlowTTS
from scipy.io.wavfile import write
import numpy as np
from src.metric import Evaluater
from hifigan.denoiser import Denoiser
from hifigan.meldataset import mel_spectrogram
from tqdm import tqdm
import soundfile as sf
import datetime as dt
import shutil
import torchaudio

global MAX_VALUE  
MAX_VALUE= 32768.0

def append_to_json(file_path, new_data):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            try:
                result_dict = json.load(file)
            except json.JSONDecodeError:
                result_dict = []
    else:
        result_dict = []

    result_dict.append(new_data)
    with open(file_path, "w") as file:
        json.dump(result_dict, file, indent=4)

@torch.inference_mode()
def synthesize(model, vocoder, denoiser, filelist, cfg, device):

    syn_path    = f'{cfg.output_folder}/syn/'
    trg_path    = f'{cfg.output_folder}/trg/'
    voc_path    = f'{cfg.output_folder}/voc/'
    os.makedirs(syn_path, exist_ok=True)
    os.makedirs(trg_path, exist_ok=True)
    os.makedirs(voc_path, exist_ok=True)
    
    meta_data = {'trg_path':[], 'syn_path':[], 'text':[], 'rtf':[], 'rtf_w':[]}
    for i, line in enumerate(tqdm(filelist)):
        wav_path, org_text, cleaned_text, spk = line

        y, sr = torchaudio.load(wav_path)
        y     = mel_spectrogram(y, cfg.preprocess.n_fft, cfg.preprocess.n_mels, cfg.preprocess.sample_rate, cfg.preprocess.hop_length, cfg.preprocess.win_length, cfg.preprocess.f_min, cfg.preprocess.f_max, center=False).to(device)
        
        x = cleaned_text_to_sequence(cleaned_text)
        if cfg.model.add_blank:
            x = torch.tensor(intersperse(x, 0), dtype=torch.long, device=device)[None]
        else:
            x = torch.tensor(x, dtype=torch.long, device=device)[None]

        x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=device)
        x_phones  = sequence_to_text(x.squeeze(0).tolist())
        spk       = torch.tensor([int(spk)],dtype=torch.long, device=device) if model.n_spks > 1 else None

        start_t = dt.datetime.now()
        output = model.synthesise(x, x_lengths, n_timesteps=cfg.n_timesteps, temperature=cfg.temperature, spks=spk, length_scale=cfg.length_scale)
        output.update({'start_t': start_t})

        output['waveform'] = vocoder(output['mel']).clamp(-1, 1)
        output['waveform'] = denoiser(output['waveform'].squeeze(0), strength=0.00025).cpu().squeeze()
        t      = (dt.datetime.now() - output['start_t']).total_seconds()
        rtf_w  = t * 22050 / (output['waveform'].shape[-1])

        y = vocoder(y).clamp(-1, 1)
        y = denoiser(y.squeeze(0), strength=0.00025).cpu().squeeze()


        spk      = spk.item() if model.n_spks > 1 else '0'
        basename = os.path.basename(wav_path).split('.')[0] + '.wav'
        syn_save_path  = f'{syn_path}/{spk}_{basename}'
        trg_save_path  = f'{trg_path}/{spk}_{basename}'
        voc_save_path  = f'{voc_path}/{spk}_{basename}'
            
        shutil.copyfile(wav_path, trg_save_path) 
        sf.write(syn_save_path, output['waveform'], 22050, 'PCM_24')
        sf.write(voc_save_path, y, 22050, 'PCM_24')

        meta_data['trg_path'].append(trg_save_path)
        meta_data['syn_path'].append(syn_save_path)
        meta_data['text'].append(org_text)  
        meta_data['rtf'].append(output['rtf'])
        meta_data['rtf_w'].append(rtf_w)

    return meta_data


def main(cfg):
    
    cfg.output_folder = os.path.join(cfg.output_folder, f'{cfg.model_name}-{cfg.n_timesteps}')
    MakeDir(cfg.output_folder)
    
    seed_init(seed=cfg.seed)   
    model = RapFlowTTS(cfg.model).to(cfg.device)
    ckpt  = torch.load(os.path.join(cfg.weight_path, f'{cfg.weight_name}.pth'), map_location=cfg.device)
    if cfg.test.ema:
        model.load_state_dict(ckpt['ema'])
    else:
        model.load_state_dict(ckpt['state_dict'], strict=False)

    evaluater = Evaluater(cfg) 
    vocoder   = get_vocoder(cfg, cfg.device)
    denoiser  = Denoiser(vocoder, mode='zeros')  ## get denoiser

    filelist = parse_filelist(cfg.path.test_path, split_char='|')
    # filelist = random.sample(filelist, k=3)

    model.eval()
    with torch.no_grad():
        meta_data = synthesize(model, vocoder, denoiser, filelist, cfg, cfg.device)
        
        cer, cer_std, wer, wer_std = evaluater.calculate_asr_score(meta_data)

    print('Test results | CER:{:.4f} | WER:{:.4f}'.format(cer, wer))
    print(f"Mean RTF: {np.mean(meta_data['rtf']):.6f} ± {np.std(meta_data['rtf']):.6f}")
    print(f"Mean RTF Waveform: {np.mean(meta_data['rtf_w']):.6f} ± {np.std(meta_data['rtf_w']):.6f}")
    
    result_dict = {"model_name":f'{cfg.model_name}-{cfg.n_timesteps}', "rtf":[np.mean(meta_data['rtf']), np.std(meta_data['rtf'])], "cer":cer, "wer":wer}
    append_to_json('./result.json', result_dict)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',        type=str,   default='cuda:0', help='Device')
    parser.add_argument('--weight_path',   type=str,   default='./checkpoints/RapFlow-TTS', help='pre-trained weight path')
    parser.add_argument('--model_name',    type=str,   default='RapFlowTTS', help='pre-trained weight path')
    parser.add_argument('--seed',          type=int,   default=100, help='seed number')
    parser.add_argument('--output_folder', type=str,   default='./output', help='seed number')
    parser.add_argument('--n_timesteps',   type=int,   default=10, help='Time step')
    parser.add_argument('--temperature',   type=float, default=0.667, help='Temperature')
    parser.add_argument('--length_scale',  type=float, default=1.0, help='length scale')
    parser.add_argument('--weight_name',   type=str,   default='model-last', help='length scale')
    
    args = parser.parse_args()
        
    cfg_path = os.path.join(args.weight_path, 'base.yaml')
    cfg  = Config(cfg_path)
    args = get_params(args)
    for key in args:
        cfg[key] = args[key]    

    print(cfg)
    main(cfg)

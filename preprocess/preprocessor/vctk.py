"""
RapFlow-TTS
Copyright (c) 2025-present NAVER Cloud Corp.
Apache-2.0
"""

import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from text import text_to_sequence
from joblib import Parallel, delayed
import resampy
import soundfile as sf
import random

def MakeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text

def get_clean_text(cfg, path='./filelists/VCTK'):

    def text_process(wav_path, text, spk, cfg):
        text_norm, cleaned_text = text_to_sequence(text, cfg['preprocessing']['text']['text_cleaners'])
        new_line = '|'.join([wav_path, text, cleaned_text, str(spk) + '\n'])
        return new_line

    for file in os.listdir(path):
        tmp_path = os.path.join(path, file)
        tmp_file = parse_filelist(tmp_path)
                
        new_files = Parallel(n_jobs=16)(delayed(text_process)(wav_path, text, spk, cfg) for wav_path, text, spk in tqdm(tmp_file))
    
        with open(os.path.join(path, 'cleaned_'+file), "w") as f:
            f.writelines(new_files)


def split_train_val_test(meta_list, config):
    
    dataset   = config["dataset"]
    wav_path  = config["path"]["wav_path"] 
    
    spk_list = sorted(os.listdir(wav_path))
    print('Number of speakers:', len(spk_list))
    spk_dict = {k:v for v,k in enumerate(spk_list)}
    print(spk_dict)
    
    filelist  = []
    for (org_path, file_path, text, speaker, base_name) in meta_list:
        spk       = str(spk_dict[speaker])
        strings   = '|'.join([file_path, text, spk + '\n'])
        filelist.append(strings)

    filelist = sorted(filelist)
    random.shuffle(filelist)
    
    val_size  = int(0.8 * len(filelist))
    test_size = int(0.9 * len(filelist))
    train_filelist = filelist[:val_size]
    val_filelist   = filelist[val_size:test_size]
    test_filelist  = filelist[test_size:]
    print(len(filelist), len(train_filelist), len(val_filelist), len(test_filelist))
    
    write_path = f'./filelists/{dataset}'
    os.makedirs(write_path, exist_ok=True)
    with open(f"{write_path}/train.txt", "w") as file:
        file.writelines(train_filelist)
    with open(f"{write_path}/valid.txt", "w") as file:
        file.writelines(val_filelist)
    with open(f"{write_path}/test.txt", "w") as file:
        file.writelines(test_filelist)
        
def save_meta(meta_info, out_path, sr, config):
    
    wav_path, out_wav_path, text, speaker, base_name = meta_info

    wav, fs = sf.read(wav_path)
    if fs != sr:
        wav = resampy.resample(x=wav, sr_orig=fs, sr_new=sr, axis=0)
    wav = wav / max(abs(wav))
    sf.write(os.path.join(out_path, speaker, "{}.wav".format(base_name)), wav, sr)        
        
    with open(os.path.join(out_path, speaker, "{}.lab".format(base_name)), "w") as f1:
        f1.write(text)
    
def prepare_align(config):
    
    wav_path  = config["path"]["wav_path"] 
    text_path = wav_path.replace('wav48', 'txt')
    out_path  = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    
    total_meta_list = [] # wav_path, text, speaker, base_name
    for speaker in tqdm(os.listdir(wav_path)):
        
        os.makedirs(os.path.join(out_path, speaker), exist_ok=True)
        for file_name in os.listdir(os.path.join(wav_path, speaker)):

            base_name     = file_name[:-4]
            tmp_text_path = os.path.join(text_path, speaker, "{}.txt".format(base_name))
            tmp_wav_path  = os.path.join(wav_path, speaker, "{}.wav".format(base_name))            
            tmp_out_path  = os.path.join(out_path, speaker, "{}.wav".format(base_name))  

            if file_name[-4:] != ".wav" or not os.path.isfile(tmp_text_path) or not os.path.isfile(tmp_wav_path):
                continue
            
            with open(tmp_text_path) as f:
                text = f.readline().strip("\n")
                
            total_meta_list.append([tmp_wav_path, tmp_out_path, text, speaker, base_name])
            
    Parallel(n_jobs=16)(delayed(save_meta)(meta_info, out_path, sampling_rate, config) for meta_info in tqdm(total_meta_list))
    
    return total_meta_list


def preprocess(config):
        
    meta_list = prepare_align(config)
    
    split_train_val_test(meta_list, config)
    
    get_clean_text(config)
    

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

def MakeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text

def get_clean_text(cfg, path='./filelists/LJSpeech'):

    for file in os.listdir(path):
        tmp_path = os.path.join(path, file)
        tmp_file = parse_filelist(tmp_path)

        new_files = []
        for line in tqdm(tmp_file):
            wav_path, text, spk = line
            text_norm, cleaned_text = text_to_sequence(text, cfg['preprocessing']['text']['text_cleaners'])

            new_line = '|'.join([wav_path, text, cleaned_text, str(spk) + '\n'])
            new_files.append(new_line)

        with open(os.path.join(path, 'cleaned_'+file), "w") as f:
            f.writelines(new_files)

def get_meta_data(config, ref_path='./resources/filelists/LJSpeech'):
    
    dataset = config['dataset']
    files   = os.listdir(ref_path)
    
    for file in files:
        
        write_path = f'./filelists/{dataset}'
        MakeDir(write_path)
        file_path = os.path.join(ref_path, file)
        tmp_file  = parse_filelist(file_path, 'No Splitter')
        
        new_files = []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                new_line = line.strip().replace('DUMMY', config['path']['wav_path']) + '|0' +'\n'
                new_files.append(new_line)
        
        with open(os.path.join(write_path, file), "w") as f:
            f.writelines(new_files)

def preprocess(config):
    
    get_meta_data(config)
    get_clean_text(config)
    
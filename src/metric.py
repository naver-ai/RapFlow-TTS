# reference: https://github.com/winddori2002/DEX-TTS

import logging
import librosa
import numpy as np

from tqdm import tqdm
from typing import Optional, Union
from pathlib import Path

import jiwer
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

class Evaluater:
    
    def __init__(self, cfg):
        
        self.cfg       = cfg
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        self.asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(cfg.device)
        
    def transcribe(self, wav_path):
        
        wav, _         = librosa.load(wav_path, sr=16000)
        input_features = self.processor(wav, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            predicted_ids = self.asr_model.generate(input_features.to(self.cfg.device))[0]
        transcription = self.processor.decode(predicted_ids)
        
        return transcription
        
    def calculate_wer_cer(self, gt, pred):
        
        gt   = self.processor.tokenizer._normalize(gt)
        pred = self.processor.tokenizer._normalize(pred)
        
        cer  = jiwer.cer(gt, pred)
        wer  = jiwer.wer(gt, pred)     
        
        return cer, wer, gt, pred
        
    def calculate_asr_score(self, meta_data):
        
        cer_list  = []
        wer_list  = []
        total_cer = 0
        total_wer = 0
        trg_paths = meta_data['trg_path']
        syn_paths = meta_data['syn_path']
        texts     = meta_data['text']
        
        gt_list        = []
        gt_norm_list   = []
        pred_list      = []
        pred_norm_list = []
        for (trg, syn, txt) in zip(trg_paths, syn_paths, texts):
                        
            pred      = self.transcribe(syn)
            cer, wer, norm_gt, norm_pred  = self.calculate_wer_cer(txt, pred)
            total_cer += cer
            total_wer += wer        
            cer_list.append(cer)
            wer_list.append(wer)
            
            gt_list.append(txt)
            gt_norm_list.append(norm_gt)
            pred_list.append(pred)
            pred_norm_list.append(norm_pred)
            

        cer_std = np.std(cer_list) / np.sqrt(len(trg_paths))
        wer_std = np.std(wer_list) / np.sqrt(len(trg_paths))
        
        return total_cer / len(trg_paths), cer_std, total_wer / len(trg_paths), wer_std
    
    def get_asr_results(self, meta_data):
        
        trg_paths = meta_data['trg_path']
        syn_paths = meta_data['syn_path']
        texts     = meta_data['text']
        new_meta  = {"gt":[], "gt_norm":[], "pred":[], "pred_norm":[], "wer":[], "cer":[]}
        for (trg, syn, txt) in zip(trg_paths, syn_paths, texts):
                        
            pred      = self.transcribe(syn)
            cer, wer, norm_gt, norm_pred  = self.calculate_wer_cer(txt, pred)
            new_meta['gt'].append(txt)
            new_meta['gt_norm'].append(norm_gt)
            new_meta['pred'].append(pred)
            new_meta['pred_norm'].append(norm_pred)
            new_meta['wer'].append(wer)
            new_meta['cer'].append(cer)

        return new_meta
    

def normalize_sentence(sentence):
    """Normalize sentence"""
    # Convert all characters to upper.
    sentence = sentence.upper()
    # Delete punctuations.
    sentence = jiwer.RemovePunctuation()(sentence)
    # Remove \n, \t, \r, \x0c.
    sentence = jiwer.RemoveWhiteSpace(replace_by_space=True)(sentence)
    # Remove multiple spaces.
    sentence = jiwer.RemoveMultipleSpaces()(sentence)
    # Remove white space in two end of string.
    sentence = jiwer.Strip()(sentence)

    # Convert all characters to upper.
    sentence = sentence.upper()

    return sentence

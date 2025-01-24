# reference: https://github.com/shivammehta25/Matcha-TTS

import random
import numpy as np

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader 
from pathlib import Path
import resampy
import soundfile as sf
from text import text_to_sequence, cleaned_text_to_sequence
from text.symbols import symbols
from src.utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility, normalize
from hifigan.meldataset import mel_spectrogram



class TextMelSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cfg):
        self.filelist    = parse_filelist(filelist_path, split_char='|')
        self.n_spks      = cfg.model.n_spks
        self.add_blank   = cfg.model.add_blank
        self.n_fft       = cfg.preprocess.n_fft
        self.n_mels      = cfg.preprocess.n_mels
        self.sample_rate = cfg.preprocess.sample_rate
        self.hop_length  = cfg.preprocess.hop_length
        self.win_length  = cfg.preprocess.win_length
        self.f_min       = cfg.preprocess.f_min
        self.f_max       = cfg.preprocess.f_max
        self.cleaners    = cfg.preprocess.cleaner
        
        self.mel_mean, self.mel_std = cfg.model.data_stats
        self.load_from_clean = cfg.train.load_from_clean 
        self.load_from_disk  = cfg.train.load_from_disk
        
        random.seed(cfg.seed)
        random.shuffle(self.filelist)
        # self.filelist = self.filelist[:100]


    def get_datapoint(self, filelist):
        
        filepath, text, spk = filelist[0], filelist[1], filelist[-1] 
        
        if self.n_spks <= 1:
            spk = None
        else:
            spk = int(spk)
        
        if self.load_from_clean:
            cleaned_text = filelist[2]
            text, cleaned_text = self.get_text(text, cleaned_text=cleaned_text)
        else:
            text, cleaned_text = self.get_text(text)
        mel = self.get_mel(filepath)

        durations = None

        return {"x": text, "y": mel, "spk": spk, "filepath": filepath, "x_text": cleaned_text, "durations": durations}


    def get_mel(self, filepath):
        
        if not self.load_from_disk:
            audio, sr = torchaudio.load(filepath)
            assert sr == self.sample_rate
            mel = mel_spectrogram(
                audio,
                self.n_fft,
                self.n_mels,
                self.sample_rate,
                self.hop_length,
                self.win_length,
                self.f_min,
                self.f_max,
                center=False,
            ).squeeze()
            
        else:
            filepath = Path(filepath)
            data_dir, name = filepath.parent.parent, filepath.stem
            mel_path = data_dir / "mels" / f"{name}.npy"
            mel      = torch.from_numpy(np.load(mel_path))
            
        mel = normalize(mel, self.mel_mean, self.mel_std)
        
        return mel

    def get_text(self, text, cleaned_text=None):
        if cleaned_text is None:
            text_norm, cleaned_text = text_to_sequence(text, self.cleaners)
        else:
            text_norm = cleaned_text_to_sequence(cleaned_text)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.IntTensor(text_norm)
        return text_norm, cleaned_text

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.filelist[index])
        return datapoint

    def __len__(self):
        return len(self.filelist)


class TextMelSpeakerBatchCollate:
    def __init__(self, n_spks):
        self.n_spks = n_spks

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])
        n_feats      = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        durations = torch.zeros((B, x_max_length), dtype=torch.long)

        y_lengths, x_lengths = [], []
        spks = []
        filepaths, x_texts = [], []
        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        spks = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None

        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "spks": spks,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "durations": durations if not torch.eq(durations, 0).all() else None,
        }

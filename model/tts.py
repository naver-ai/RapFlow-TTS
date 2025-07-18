import datetime as dt
import math
import random

import torch
import torch.nn as nn
from model import monotonic_align
from model.flow_matching import ConsistencyFM
from model.text_encoder import TextEncoder
from model.utils import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)


class RapFlowTTS(nn.Module):
    def __init__(self, cfg):
        super(RapFlowTTS, self).__init__()
        
        self.n_feats    = cfg.n_feats
        self.n_spks     = cfg.n_spks
        self.prior_loss = cfg.prior_loss
        
        if cfg.n_spks > 1:
            self.spk_emb = torch.nn.Embedding(cfg.n_spks, cfg.spk_emb_dim)
        
        self.encoder = TextEncoder(cfg.encoder.encoder_type, cfg.encoder.encoder_params, cfg.encoder.dp_params, 
                                   cfg.n_vocab, cfg.n_spks, cfg.spk_emb_dim,)

        self.decoder = ConsistencyFM(in_channels=2 * cfg.encoder.encoder_params.n_feats, out_channel=cfg.encoder.encoder_params.n_feats, 
                           cfm_params=cfg.cfm, decoder_params=cfg.decoder, n_spks=cfg.n_spks, spk_emb_dim=cfg.spk_emb_dim)

        self.mel_mean, self.mel_std = cfg.data_stats
        
    # adapated from https://github.com/shivammehta25/Matcha-TTS
    @torch.inference_mode()
    def synthesise(self, x, x_lengths, n_timesteps, temperature=0.667, spks=None, length_scale=1.0):

        t = dt.datetime.now()
        
        if self.n_spks > 1:
            spks = self.spk_emb(spks.long())

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)

        w             = torch.exp(logw) * x_mask
        w_ceil        = torch.ceil(w) * length_scale
        y_lengths     = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length  = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask    = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn      = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Generate sample tracing the probability flow
        trajectories    = self.decoder(mu_y, y_mask, n_timesteps, temperature, spks)
        decoder_outputs = trajectories[-1]
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length],
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths,
            "rtf": rtf,
            "traj": trajectories
        }

    # adapated from https://github.com/shivammehta25/Matcha-TTS, https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS
    def forward(self, x, x_lengths, y, y_lengths, spks=None, out_size=None, cond=None, durations=None, cur_epoch=None):

        if self.n_spks > 1:
            # Get speaker embedding
            spks = self.spk_emb(spks)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)
        y_max_length       = y.shape[-1]

        y_mask    = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)


        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const       = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor      = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square    = torch.matmul(factor.transpose(1, 2), y**2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square   = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
            log_prior   = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()  # b, t_text, T_mel

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        # refered to as prior loss in the paper
        logw_    = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)


        if not isinstance(out_size, type(None)):
            max_offset    = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset    = torch.LongTensor(
                [torch.tensor(random.choice(range(start, end)) if end > start else 0) for start, end in offset_ranges]
            ).to(y_lengths)
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut    = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask    = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn   = attn_cut
            y      = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of the decoder
        diff_loss, x1, x1_pred = self.decoder.compute_consistency_fm_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=cond, cur_epoch=cur_epoch)

        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0

        return dur_loss, prior_loss, diff_loss, x1, x1_pred
    

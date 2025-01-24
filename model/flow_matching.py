"""
RapFlow-TTS
Copyright (c) 2025-present NAVER Cloud Corp.
Apache-2.0
"""

from abc import ABC

import torch
import torch.nn.functional as F
import numpy as np
import math

from model.decoder import Decoder
from model.utils import *
import matplotlib.pyplot as plt

class BaseConsistencyFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks      = n_spks
        self.spk_emb_dim = spk_emb_dim
        
        self.alpha          = float(cfm_params.alpha)      # 1e-5 --> velocity space consistency ratio
        self.boundary       = float(cfm_params.boundary)
        self.total_epoch    = cfm_params.total_epoch

        self.delta_init     = float(cfm_params.delta_t[0])
        self.delta_end      = float(cfm_params.delta_t[1])
        self.delta_bin      = cfm_params.delta_bin
        
        # 0 (stage1) -> 0.9 (stage2) time threshold ratio
        if cfm_params.delta_schedule == 'linear':
            schedule = torch.linspace(self.delta_init, self.delta_end, self.delta_bin)
            bin_cnt  = int(self.total_epoch / self.delta_bin)
            re_cnt   = self.total_epoch - (self.delta_bin * bin_cnt)
            schedule = schedule.repeat_interleave(bin_cnt)
            schedule = torch.cat([schedule, schedule[-1].repeat(re_cnt)])
            self.delta_schedule = schedule
        
        self.use_d          = cfm_params.use_d_schedule
        self.loss_type      = cfm_params.loss_type

        self.num_segments   = cfm_params.num_segments      # 2 multi-segment
        self.total_epoch    = cfm_params.total_epoch
        self.estimator      = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):

        z      = torch.randn_like(mu) * temperature
        return self.solve_euler_cfm(z, n_timesteps=n_timesteps, mu=mu, mask=mask, spks=spks, cond=cond)
    
    def compute_loss(self, x, y, mask, app_mask=None, c=0.00054):
        
        if self.loss_type == 'huber':
            c    = c * math.sqrt(math.prod(x.shape[1:]))
            loss = torch.sqrt((x - y)**2 + c**2) - c
        else:
            loss = F.mse_loss(x, y, reduction="none") 
            
        if app_mask is not None:
            # mask = mask * app_mask
            loss = loss * app_mask
            
        mask = mask.bool()
        loss = loss.masked_select(mask).mean() 
        
        return loss
    
    def solve_euler_cfm(self, x, n_timesteps, mu, mask, spks, cond):

        dt           = 1. / n_timesteps
        eps          = 1e-3 # default: 1e-3
        
        traj = [x]
        for i in range(n_timesteps):
            
            num_t = i / n_timesteps * (1 - eps) + eps
            t     = torch.ones(x.shape[0], device=x.device) * num_t
            pred  = self.estimator(x, mask, mu, t, spks, cond) 
            x     = x.detach().clone() + pred * dt
            traj.append(x)
        
        return traj
    
    # adapted from https://github.com/YangLing0818/consistency_flow_matching
    def compute_consistency_fm_loss(self, x1, mask, mu, eps=1e-3, spks=None, cond=None, cur_epoch=None):

        b, _, t = mu.shape
        
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        # sample random timestep (t and t + delta_t)            
        if self.use_d:
            delta_t = self.delta_schedule[cur_epoch-1]
        else:
            delta_t = self.delta_end
        
        t = torch.rand([b], device=mu.device, dtype=mu.dtype)
        t = t * (1 - eps) + eps
        r = torch.clamp(t + delta_t, max=1.0)
        
        # sample interploation xt = x1 * t + (1 - t) * x0
        t_expand = t.view(-1, 1, 1).repeat(1, x1.shape[1], x1.shape[2])
        r_expand = r.view(-1, 1, 1).repeat(1, x1.shape[1], x1.shape[2])
        xt       = t_expand * x1 + (1. - t_expand) * z
        xr       = r_expand * x1 + (1. - r_expand) * z
        
        segments     = torch.linspace(0, 1, self.num_segments + 1, device=x1.device)  # multi_segment (0~0.5, 0.5~1)
        seg_indices  = torch.searchsorted(segments, t, side="left").clamp(min=1) # .clamp(min=1) prevents the inclusion of 0 in indices.
        segment_ends = segments[seg_indices]  # multi-segment ends  0.5 or 1.0
        
        segment_ends_expand = segment_ends.view(-1, 1, 1).repeat(1, x1.shape[1], x1.shape[2])
        x_at_segment_ends   = segment_ends_expand * x1 + (1. - segment_ends_expand) * z
                
        def f_euler(t_expand, segment_ends_expand, xt, vt):
            return xt + (segment_ends_expand - t_expand) * vt
        def threshold_based_f_euler(t_expand, segment_ends_expand, xt, vt, threshold, x_at_segment_ends):
            if threshold == 0:  
                return x_at_segment_ends
            
            less_than_threshold = t_expand < threshold  ##  thresh=0.9 
            
            # gt or consistency
            res = (
                less_than_threshold * f_euler(t_expand, segment_ends_expand, xt, vt) # 
                + (~less_than_threshold) * x_at_segment_ends  
                )
            return res
        
        t         = t.view(-1, 1, 1)        
        r         = r.view(-1, 1, 1)
        rng_state = torch.cuda.get_rng_state()
        vt        = self.estimator(xt, mask, mu, t.squeeze(), spks) ## vt = v(xt, t)

        torch.cuda.set_rng_state(rng_state) # Shared Dropout Mask
        with torch.no_grad():
            if self.boundary == 0: # boundary == 0 --> stage 1, (straight flow)
                vr = None
            else:
                vr = self.estimator(xr, mask, mu, r.squeeze(), spks) ## vr = v(xr, r), (straight flow & consistency loss)
                vr = torch.nan_to_num(vr)
        
        # velocity --> sample space
        ft = f_euler(t_expand, segment_ends_expand, xt, vt)
        fr = threshold_based_f_euler(r_expand, segment_ends_expand, xr, vr, self.boundary, x_at_segment_ends)

        ##### f loss #####
        loss_f = self.compute_loss(ft, fr, mask, app_mask=None)
        
        def masked_loss_v(vt, vr, threshold, segment_ends, t):
            if threshold == 0: 
                return 0
            
            less_than_threshold   = t_expand < threshold
            far_from_segment_ends = (segment_ends - t) > 1.01 * delta_t
            far_from_segment_ends = far_from_segment_ends.view(-1, 1, 1).repeat(1, x1.shape[1], x1.shape[2])
            
            app_mask = less_than_threshold * far_from_segment_ends
            loss_v   = self.compute_loss(vt, vr, mask, app_mask=app_mask)

            return loss_v

        #### loss #####
        loss_v = masked_loss_v(vt, vr, self.boundary, segment_ends, t.squeeze().squeeze())
        loss   = loss_f + self.alpha * loss_v  
        
        x1      = x_at_segment_ends * mask
        x1_pred = ft * mask
        
        return loss, x1, x1_pred


# reference https://github.com/shivammehta25/Matcha-TTS
class ConsistencyFM(BaseConsistencyFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        in_channels    = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
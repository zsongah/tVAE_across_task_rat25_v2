# models/LFADS.py
import torch
from torch import nn, Tensor
import numpy as np
from typing import Dict, Tuple, List, Union, Any

from .LFADS_modules.encoder import Encoder
from .LFADS_modules.decoder import Decoder
from .LFADS_modules.priors import MultivariateNormal,AutoregressiveMultivariateNormal
from .LFADS_modules.l2 import compute_l2_penalty

class LFADS(nn.Module):
    def __init__(self, config, device, model_type):
        super(LFADS, self).__init__()
        self.model_type = model_type
        self.config = config
        self.device = device 
        # networks
        self.readin = nn.Identity()
        self.readout = nn.Linear(config.MODEL.LATENT_DIM, config.MODEL.ENCOD_DATA_DIM)
        self.encoder = Encoder(config, device)
        self.decoder = Decoder(config, device)
        self.ic_prior = MultivariateNormal(mean=0,variance=1,shape=config.MODEL.IC_DIM)
        self.co_prior = AutoregressiveMultivariateNormal(tau=10,nvar=0.1,shape=config.MODEL.CO_DIM)
        self.use_con = all([config.MODEL.CI_ENC_DIM > 0, config.MODEL.CON_DIM > 0, config.MODEL.CO_DIM > 0])
        self.current_epoch = 0
    def forward(
            self,
            batch: Tensor,
            sample_posteriors: bool = False):
        # batch: [batch_size, seq_len, input_dim]
        batch_size = batch.shape[0]
        encod_data = self.readin(batch)
        ic_mean, ic_std, ci = self.encoder(encod_data)
        ic_post = self.ic_prior.make_posterior(ic_mean, ic_std)
        ic_samp = ic_post.rsample() if sample_posteriors else ic_mean
        ext_input = torch.zeros(batch_size, self.config.MODEL.ENCOD_SEQ_LEN, 0).to(self.device)
        (
            gen_init,
            gen_states,
            con_states,
            co_means,
            co_stds,
            gen_inputs,
            factors,
        ) = self.decoder(ic_samp, ci, ext_input, sample_posteriors=sample_posteriors)
        pred = self.readout(factors)
        output = {
            'pred': pred,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ci': ci,
            'gen_init': gen_init,
            'gen_states': gen_states,
            'con_states': con_states,
            'co_means': co_means,
            'co_stds': co_stds,
            'gen_inputs': gen_inputs,
            'factors': factors,
        }
        return output

    def loss_function(
            self, 
            batch: Tensor,
            truth: Tensor):
        output = self.forward(batch)
        if self.training:
            pred = output['pred'].reshape(-1, self.config.MODEL.ENCOD_DATA_DIM)
        else:
            pred = output['pred'][:,-self.config.TRAIN.STEP_SIZE_TEST:,:].reshape(-1, self.config.MODEL.ENCOD_DATA_DIM)
        # compute the reconstruction loss
        recon_loss = nn.functional.mse_loss(pred, truth, reduction='mean')
        # compute the L2 penalty on recurrent weights
        l2 = self.compute_l2_penalty()
        
        ic_mean = output['ic_mean']
        ic_std = output['ic_std']
        co_means = output['co_means']
        co_stds = output['co_stds']
        ic_kl = self.ic_prior(ic_mean, ic_std) * self.config.TRAIN.KL_IC_SCALE
        co_kl = self.co_prior(co_means, co_stds) * self.config.TRAIN.KL_CO_SCALE
        l2_ramp = self._compute_ramp(self.config.TRAIN.L2_START_EPOCH, self.config.TRAIN.L2_INCREASE_EPOCH)
        kl_ramp = self._compute_ramp(self.config.TRAIN.KL_START_EPOCH, self.config.TRAIN.KL_INCREASE_EPOCH)
        loss = self.config.LOSS_SCALE * (recon_loss + l2_ramp * l2 + kl_ramp * (ic_kl + co_kl))

        return loss, self.config.LOSS_SCALE *recon_loss, self.config.LOSS_SCALE *l2, self.config.LOSS_SCALE *(ic_kl + co_kl) 
    
    def _compute_ramp(self, start: int, increase: int):
        # Compute a coefficient that ramps from 0 to 1 over `increase` epochs
        ramp = (self.current_epoch + 1 - start) / (increase + 1)
        return torch.clamp(torch.tensor(ramp), 0, 1)
    
    def compute_l2_penalty(self):
        recurrent_kernels_and_weights = [
        (self.encoder.ic_enc.fwd_gru.cell.weight_hh, self.config.TRAIN.L2_IC_ENC_SCALE),
        (self.encoder.ic_enc.bwd_gru.cell.weight_hh, self.config.TRAIN.L2_IC_ENC_SCALE),
        (self.decoder.rnn.cell.gen_cell.weight_hh, self.config.TRAIN.L2_GEN_SCALE),
        ]
        if self.use_con:
            recurrent_kernels_and_weights.extend(
                [
                    (self.encoder.ci_enc.fwd_gru.cell.weight_hh, self.config.TRAIN.L2_CI_ENC_SCALE),
                    (self.encoder.ci_enc.bwd_gru.cell.weight_hh, self.config.TRAIN.L2_CI_ENC_SCALE),
                    (self.decoder.rnn.cell.con_cell.weight_hh, self.config.TRAIN.L2_CON_SCALE),
                ]
            )
        # Add recurrent penalty
        recurrent_penalty = torch.tensor(0.0, device=self.device)
        recurrent_size = 0
        for kernel, weight in recurrent_kernels_and_weights:
            if weight > 0:
                recurrent_penalty += weight * 0.5 * torch.norm(kernel, 2) ** 2
                recurrent_size += kernel.numel()
        recurrent_penalty /= recurrent_size + 1e-8

        return recurrent_penalty
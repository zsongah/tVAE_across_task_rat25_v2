import torch
import torch.nn.functional as F
from torch import nn

from .initializers import init_linear_
from .recurrent import BidirectionalClippedGRU


class Encoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        # Initial hidden state for IC encoder
        self.ic_enc_h0 = nn.Parameter(
            torch.zeros((2, 1, config.MODEL.IC_ENC_DIM), requires_grad=True)
        )
        # Initial condition encoder
        self.ic_enc = BidirectionalClippedGRU(
            input_size=config.MODEL.ENCOD_DATA_DIM,
            hidden_size=config.MODEL.IC_ENC_DIM, 
            clip_value=config.CELL_CLIP,
        )
        # Mapping from final IC encoder state to IC parameters
        self.ic_linear = nn.Linear(config.MODEL.IC_ENC_DIM * 2, config.MODEL.IC_ENC_DIM * 2)
        init_linear_(self.ic_linear)
        # Decide whether to use the controller
        self.use_con = all(
            [
                config.MODEL.CI_ENC_DIM > 0, # Dimensionality of the controller input encoder 
                config.MODEL.CON_DIM > 0,
                config.MODEL.CO_DIM > 0,
            ]
        )
        if self.use_con:
            # Initial hidden state for CI encoder
            self.ci_enc_h0 = nn.Parameter(
                torch.zeros((2, 1, config.MODEL.CI_ENC_DIM), requires_grad=True)
            )
            # CI encoder
            self.ci_enc = BidirectionalClippedGRU(
                input_size=config.MODEL.ENCOD_DATA_DIM,
                hidden_size=config.MODEL.CI_ENC_DIM,
                clip_value=config.CELL_CLIP,
            )
        # Activation dropout layer
        self.dropout = nn.Dropout(config.DROPOUT_RATE)

    def forward(self, data: torch.Tensor):

        batch_size = data.shape[0]
        assert data.shape[1] == self.config.MODEL.ENCOD_SEQ_LEN, (
            f"Sequence length specified in config ({self.config.ENCOD_SEQ_LEN}) "
            f"must match data dim 1 ({data.shape[1]})."
        )
        data_drop = self.dropout(data)
        # option to use separate segment for IC encoding
        ic_enc_data = data_drop
        ci_enc_data = data_drop
        # Pass data through IC encoder
        ic_enc_h0 = torch.tile(self.ic_enc_h0, (1, batch_size, 1))
        _, h_n = self.ic_enc(ic_enc_data, ic_enc_h0)
        h_n = torch.cat([*h_n], dim=1) # merge forward and backward hidden states
        # Compute initial condition posterior
        h_n_drop = self.dropout(h_n)
        ic_params = self.ic_linear(h_n_drop)
        ic_mean, ic_logvar = torch.split(ic_params, self.config.MODEL.IC_DIM, dim=1)
        ic_std = torch.sqrt(torch.exp(ic_logvar) + self.config.IC_POST_VAR_MIN)
        if self.use_con:
            # Pass data through CI encoder
            ci_enc_h0 = torch.tile(self.ci_enc_h0, (1, batch_size, 1))
            ci, _ = self.ci_enc(ci_enc_data, ci_enc_h0)
            # Add a lag to the controller input
            ci_fwd, ci_bwd = torch.split(ci, self.config.MODEL.IC_ENC_DIM, dim=2)
            ci_fwd = F.pad(ci_fwd, (0, 0, self.config.MODEL.CI_LAG, 0, 0, 0))
            ci_bwd = F.pad(ci_bwd, (0, 0, 0, self.config.MODEL.CI_LAG, 0, 0))
            ci_len = self.config.MODEL.ENCOD_SEQ_LEN - self.config.MODEL.IC_ENC_SEQ_LEN
            ci = torch.cat([ci_fwd[:, :ci_len, :], ci_bwd[:, -ci_len:, :]], dim=2)
            # Add extra zeros if necessary for forward prediction
            fwd_steps = self.config.MODEL.RECON_SEQ_LEN - self.config.MODEL.ENCOD_SEQ_LEN
            ci = F.pad(ci, (0, 0, 0, fwd_steps, 0, 0))
            # Add extra zeros if encoder does not see whole sequence
            ci = F.pad(ci, (0, 0, self.config.MODEL.IC_ENC_SEQ_LEN, 0, 0, 0))
        else:
            # Create a placeholder if there's no controller
            ci = torch.zeros(data.shape[0],  self.config.MODEL.RECON_SEQ_LEN, 0).to(data.device)

        return ic_mean, ic_std, ci

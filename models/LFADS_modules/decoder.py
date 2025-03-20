import torch
import torch.nn.functional as F
from torch import nn

from .initializers import init_linear_
from .recurrent import ClippedGRUCell
from .priors import MultivariateNormal, AutoregressiveMultivariateNormal

class KernelNormalizedLinear(nn.Linear):
    def forward(self, input):
        normed_weight = F.normalize(self.weight, p=2, dim=1)
        return F.linear(input, normed_weight, self.bias)


class DecoderCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Create the generator
        self.gen_cell = ClippedGRUCell(
            config.MODEL.EXT_INPUT_DIM + config.MODEL.CO_DIM, config.MODEL.GEN_DIM, clip_value=config.CELL_CLIP
        )
        # Create the mapping from generator states to factors
        self.fac_linear = KernelNormalizedLinear(config.MODEL.GEN_DIM, config.MODEL.LATENT_DIM, bias=False)
        init_linear_(self.fac_linear)
        # Create the dropout layer
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        # Decide whether to use the controller
        self.use_con = all(
            [
                config.MODEL.CI_ENC_DIM > 0, # Dimensionality of the controller input encoder 
                config.MODEL.CON_DIM > 0,
                config.MODEL.CO_DIM > 0,
            ]
        )
        if self.use_con:
            # Create the controller
            self.con_cell = ClippedGRUCell(
                2 * config.MODEL.CI_ENC_DIM + config.MODEL.LATENT_DIM, config.MODEL.CON_DIM, clip_value=config.CELL_CLIP
            )
            # Define the mapping from controller state to controller output parameters
            self.co_linear = nn.Linear(config.MODEL.CON_DIM, config.MODEL.CO_DIM * 2)
            init_linear_(self.co_linear)
        # Keep track of the state dimensions
        self.state_dims = [
            config.MODEL.GEN_DIM,
            config.MODEL.CON_DIM,
            config.MODEL.CO_DIM,
            config.MODEL.CO_DIM,
            config.MODEL.CO_DIM + config.MODEL.EXT_INPUT_DIM,
            config.MODEL.LATENT_DIM,
        ]
        # Keep track of the input dimensions
        self.input_dims = [2 * config.MODEL.CI_ENC_DIM, config.MODEL.EXT_INPUT_DIM]
        self.co_prior = AutoregressiveMultivariateNormal(tau=10.0, nvar=0.1, shape=config.MODEL.CO_DIM)

    def forward(self, input, h_0, sample_posteriors=True):
        config = self.config

        # Split the state up into variables of interest
        gen_state, con_state, co_mean, co_std, gen_input, factor = torch.split(
            h_0, self.state_dims, dim=1
        )
        ci_step, ext_input_step = torch.split(input, self.input_dims, dim=1)

        if self.use_con:
            # Compute controller inputs with dropout
            con_input = torch.cat([ci_step, factor], dim=1)
            con_input_drop = self.dropout(con_input)
            # Compute and store the next hidden state of the controller
            con_state = self.con_cell(con_input_drop, con_state)
            # Compute the distribution of the controller outputs at this timestep
            co_params = self.co_linear(con_state)
            co_mean, co_logvar = torch.split(co_params, config.MODEL.CO_DIM, dim=1)
            co_std = torch.sqrt(torch.exp(co_logvar))
            # Sample from the distribution of controller outputs
            co_post = self.co_prior.make_posterior(co_mean, co_std)
            con_output = co_post.rsample() if sample_posteriors else co_mean
            # Combine controller output with any external inputs
            gen_input = torch.cat([con_output, ext_input_step], dim=1)
        else:
            # If no controller is being used, can still provide ext inputs
            gen_input = ext_input_step
        # compute and store the next
        gen_state = self.gen_cell(gen_input, gen_state)
        gen_state_drop = self.dropout(gen_state)
        factor = self.fac_linear(gen_state_drop)

        hidden = torch.cat(
            [gen_state, con_state, co_mean, co_std, gen_input, factor], dim=1
        )

        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cell = DecoderCell(config=config)

    def forward(self, input, h_0, sample_posteriors=True):
        hidden = h_0
        input = torch.transpose(input, 0, 1)
        output = []
        for input_step in input:
            hidden = self.cell(input_step, hidden, sample_posteriors=sample_posteriors)
            output.append(hidden)
        output = torch.stack(output, dim=1)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        # Create the mapping from ICs to gen_state
        self.ic_to_g0 = nn.Linear(config.MODEL.IC_DIM, config.MODEL.GEN_DIM)
        init_linear_(self.ic_to_g0)
        # Create the decoder RNN
        self.rnn = DecoderRNN(config=config)
        # Initial hidden state for controller
        self.con_h0 = nn.Parameter(torch.zeros((1, config.MODEL.CON_DIM), requires_grad=True))

    def forward(self, ic_samp, ci, ext_input, sample_posteriors=True):

        # Get size of current batch (may be different than hps.batch_size)
        batch_size = ic_samp.shape[0]
        # Calculate initial generator state and pass it to the RNN with dropout rate
        gen_init = self.ic_to_g0(ic_samp)
        gen_init_drop = self.dropout(gen_init)
        # Pad external inputs if necessary and perform dropout
        fwd_steps = self.config.MODEL.RECON_SEQ_LEN - ext_input.shape[1]
        if fwd_steps > 0:
            pad = torch.zeros(batch_size, fwd_steps, self.config.MODEL.EXT_INPUT_DIM)
            ext_input = torch.cat([ext_input, pad.to(ext_input.device)], axis=1)
        ext_input_drop = self.dropout(ext_input)
        # Prepare the decoder inputs and and initial state of decoder RNN
        dec_rnn_input = torch.cat([ci, ext_input_drop], dim=2)
        device = gen_init.device
        dec_rnn_h0 = torch.cat(
            [
                gen_init,
                torch.tile(self.con_h0, (batch_size, 1)),
                torch.zeros((batch_size, self.config.MODEL.CO_DIM), device=device),
                torch.ones((batch_size, self.config.MODEL.CO_DIM), device=device),
                torch.zeros(
                    (batch_size, self.config.MODEL.CO_DIM + self.config.MODEL.EXT_INPUT_DIM), device=device
                ),
                self.rnn.cell.fac_linear(gen_init_drop),
            ],
            dim=1,
        )
        states, _ = self.rnn(
            dec_rnn_input, dec_rnn_h0, sample_posteriors=sample_posteriors
        )
        split_states = torch.split(states, self.rnn.cell.state_dims, dim=2)
        dec_output = (gen_init, *split_states)

        return dec_output

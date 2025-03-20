import torch


def compute_l2_penalty(lfads, config):
    recurrent_kernels_and_weights = [
        (lfads.encoder.ic_enc.fwd_gru.cell.weight_hh, config.TRAIN.L2_IC_ENC_SCALE),
        (lfads.encoder.ic_enc.bwd_gru.cell.weight_hh, config.TRAIN.L2_IC_ENC_SCALE),
        (lfads.decoder.rnn.cell.gen_cell.weight_hh, config.TRAIN.L2_GEN_SCALE),
    ]
    if lfads.use_con:
        recurrent_kernels_and_weights.extend(
            [
                (lfads.encoder.ci_enc.fwd_gru.cell.weight_hh, config.TRAIN.L2_CI_ENC_SCALE),
                (lfads.encoder.ci_enc.bwd_gru.cell.weight_hh, config.TRAIN.L2_CI_ENC_SCALE),
                (lfads.decoder.rnn.cell.con_cell.weight_hh, config.TRAIN.L2_CON_SCALE),
            ]
        )
    # Add recurrent penalty
    recurrent_penalty = 0.0
    recurrent_size = 0
    for kernel, weight in recurrent_kernels_and_weights:
        if weight > 0:
            recurrent_penalty += weight * 0.5 * torch.norm(kernel, 2) ** 2
            recurrent_size += kernel.numel()
    recurrent_penalty /= recurrent_size + 1e-8

    return recurrent_penalty

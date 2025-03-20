from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional
from models.stVAE_modules import Encoder, Decoder, Conv_Encoder 


class VAE(nn.Module): # nn.module is the base class for all neural network modules in pytorch
    # VAE include a Encoder and a Decoder
    def __init__(self, config, device, in_neuron_num: int, out_neuron_num: int, model_type = 'stVAE'):
        super(VAE, self).__init__()
        self.model_type = model_type
        self.config = config
        self.device = device # 每一个模块都有一个device，最开始从dataset中传入
        self.in_neuron_num = in_neuron_num
        self.out_neuron_num = out_neuron_num
        latent_dim = config.MODEL.LATENT_DIM
        if model_type == 'tVAE':
            self.encoder = Encoder(config, device, in_neuron_num) #定义了模型结构和forward函数
        elif model_type == 'stVAE':
            self.encoder = Conv_Encoder(config, device, in_neuron_num)
        self.decoder = Decoder(config, device, out_neuron_num, self.config.MODEL.DECODER_POS) #定义了模型结构和forward函数

        # latent smooth
        sigma = 1 # previous  10
        kernel_size = 2 * 4 * sigma + 1
        self.padding_size = 4 * sigma
        kernel = get_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0).repeat(latent_dim, 1, 1)
        self.register_buffer('kernel', kernel) # so that pe will not be considered as weights

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, src: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # if src_mask is None:  # TODO: use a mask to control the number of points of look back? use triu maybe
        #     """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        #     Unmasked positions are filled with float(0.0).
        #     """
        #     src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[0]).to(self.device)

        src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[0]).to(self.device)
            
        mu, log_var = self.encoder(src, src_mask)
        if self.training:  # do not sample in testing
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        return self.decoder(z, src_mask), mu, log_var

    def loss_function(self, recon_x, x, mu, log_var, beta):
        # bce = nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
        mse = nn.functional.mse_loss(recon_x, x, reduction='mean') # mean square error
        # poisson_loss = nn.functional.poisson_nll_loss(recon_x, x)
        var_priori = torch.ones_like(log_var) * 1  # restrict the variance to be 1
        # mu_priori = mu  # do not restrict the mean

        if mu.shape[0]<=self.padding_size:
            mu_priori = mu
        else:
            mu_priori = nn.functional.conv1d(
                nn.functional.pad(mu.detach().permute(1, 2, 0), (self.padding_size, self.padding_size), mode='reflect'),
                self.kernel,
                padding='valid', groups=mu.size(2)).permute(2, 0, 1)
            mu_priori *= 0.999 

        kld = -0.5 * torch.mean(
            1 + log_var - var_priori.log() - log_var.exp() / var_priori - (mu - mu_priori).pow(2) / var_priori)

        total_loss = mse + beta * kld 

        return total_loss, mse, kld

    def freeze_parameters(self,trainable_prefixes=None):
        if trainable_prefixes is None:
            trainable_prefixes = []
        for name, param in self.named_parameters():
            # if 'encoder.transformer_encoder' in name or 'decoder.transformer_decoder' in name: # 冻结encoder-decoder
            if any(name.startswith(prefix) for prefix in trainable_prefixes):
                param.requires_grad = True
            else:
                param.requires_grad = False
    
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.constant_(m.weight, 0)  # 让所有线性层的权重初始化为0
        if m.bias is not None: # 有些模型没有bias
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm): # 最开始pytorch默认参数也是这个
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    # elif isinstance(m, nn.MultiheadAttention):  # 针对 MultiheadAttention 的处理
    #     if m.in_proj_weight is not None:
    #         nn.init.xavier_uniform_(m.in_proj_weight)
    #         # nn.init.constant_(m.in_proj_weight, 0) # attention里面的qkv权重初始化为0
    #     if m.in_proj_bias is not None:
    #         nn.init.constant_(m.in_proj_bias, 0)

# 定义新的初始化函数
def initialize_weights_named(model):
     # 1. 存储初始化前的参数
    print(model)
    params_before = {}
    for name, param in model.named_parameters():
        params_before[name] = param.detach().cpu().clone()

    model.apply(initialize_weights)    
     # 3. 存储初始化后的参数
    params_after = {}
    for name, param in model.named_parameters():
        params_after[name] = param.detach().cpu().clone()
     # 4. 对比并打印参数变化
    print("\n=== 参数初始化对比 ===")
    modified_params = 0
    for name in params_before:
        before = params_before[name]
        after = params_after[name]
        if torch.equal(before, after):
            print(f"参数 '{name}' 未被修改。")
        else:
            modified_params += 1
            # 计算差异统计量
            diff = after - before
            mean_change = diff.mean().item()
            std_change = diff.std().item()
            max_change = diff.abs().max().item()
            print(f"参数 '{name}' 已被修改。差异统计：均值变化={mean_change:.6f}, 标准差变化={std_change:.6f}, 最大变化={max_change:.6f}")
    print(f"\n总参数数量: {len(params_before)}")
    print(f"被修改的参数数量: {modified_params}")

def get_kernel(size: int, sigma: float):
    # x = torch.arange(-size // 2 + 1., size // 2 + 1.).cuda()
    x = torch.arange(-size // 2 + 1., size // 2 + 1.)
    x = x / sigma
    kernel = torch.exp(-0.5 * x ** 2)
    kernel = kernel / kernel.sum()
    return kernel
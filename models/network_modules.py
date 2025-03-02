from typing import Tuple, Union
import math
import torch
from torch import nn, Tensor
import torch.nn.functional
from typing import List

class Conv_Encoder(nn.Module):

    def __init__(self, config, device, neuron_num: int):
        super(Conv_Encoder, self).__init__()
        self.model_type = 'Conv_Dynamic_Encoder'
        self.device = device
        self.d_model = config.MODEL.EMBED_DIM 
        self.neuron_num = neuron_num
        d_model = config.MODEL.EMBED_DIM
        dropout = config.MODEL.DROPOUT
        n_head = config.MODEL.NUM_HEADS
        d_hid = config.MODEL.HIDDEN_SIZE
        n_layers_encoder = config.MODEL.N_LAYERS_ENCODER
        latent_dim = config.MODEL.LATENT_DIM
        ################################# add convluational layer ################################
        kernel_sizes = [3,4,5]  # 比如通道方向的卷积核分别是高3、4、5
        out_channels = 2
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,        # 因为我们会 reshape 到 (batch, 1, channel, length)
                out_channels=out_channels,
                kernel_size=(k, 1),   # 只在通道方向(k)滑动; 时间维使用1
                stride=(1, 1),
                padding=(0, 0)       # 不做 padding，让输出在“channel”维度变小
            )
            for k in kernel_sizes
        ])
        # (2) 后续映射到 d_model 大小
        flattened_features = sum([
        out_channels * (neuron_num - k + 1) for k in kernel_sizes
    ])
        self.temp_fc = nn.Linear(
            flattened_features,
            d_model,
            bias=False
        )
        ################################ add convluational layer end #################################
        # Preprocess
        # self.embedding = nn.Linear(neuron_num, d_model, bias=False) # 加conv之前存在
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=n_head, 
                                                   dim_feedforward = d_hid,
                                                    dropout = dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers_encoder)

        # Latent space
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_var = nn.Linear(d_model, latent_dim) # need_var:


    def forward(self, src: Tensor, src_mask: Tensor = None) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)
        ################################# add convluational layer ################################
        # src:  [seq_len, batch, channel]
        # reshape: (batch, channel, seq_len)
        src = src.permute(1,2,0).unsqueeze(1)
        conv_outputs = []
        for conv in self.convs:
            # 卷积输出： (batch, out_channels, channel-k+1, length)
            c_out = conv(src)
            # 这里可选加一个激活函数，比如 ReLU
            c_out = torch.sigmoid(c_out)
            # 把 (batch, out_channels, channel_k, length) reshape/flatten，把 out_channels 和 channel_k 合并
            # 得到 (batch, out_channels*channel_k, length)
            b, oc, ck, le = c_out.shape  # ck = channel - kernel_size + 1
            c_out = c_out.view(b, oc * ck, le)
            conv_outputs.append(c_out)
         #  (2) 拼接不同卷积核的输出: (batch, sum(oc*ck), length)
        # ---------------------------------------------------------------------
        x_cat = torch.cat(conv_outputs, dim=1)  # 在channel维上拼，dim=1
        # ---------------------------------------------------------------------
        #  (3) 再把它 permute 到 (batch, length, sum(oc*ck))，投射到了 d_model
        # ---------------------------------------------------------------------
        x_cat = x_cat.permute(2, 0, 1)   # (length, batch, sum_of_channels)
        src = self.temp_fc(x_cat)     # (length, batch, d_model)
        ################################# add convluational layer end ################################
        # src: [seq_len, batch, n_channels]
        # src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2) #[batch, seq_len, n_channels]
        encoded_src = self.transformer_encoder(src, src_mask)
        encoded_src = encoded_src.permute(1, 0, 2) 
        mu = self.fc_mu(encoded_src)
        # 应用BatchNorm1d
        # mu = self.bn(mu.permute(0, 2, 1)).permute(0, 2, 1)
        log_var = self.fc_var(encoded_src)

        return mu, log_var

    def get_attention_matrix(self, src: Tensor, src_mask: Tensor = None) -> List[Tensor]:
        """
        新增函数：在不破坏原 forward 的情况下，
        手动逐层调用 TransformerEncoderLayer，收集多头注意力权重。
        
        返回一个列表，其中每个元素对应一层的注意力权重：
        attention_matrices[layer_idx] 的形状是 [batch_size, n_head, seq_len, seq_len]
        """
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)

        # -------------- 卷积 --------------
        # 跟 forward 同样的预处理
        src = src.permute(1, 2, 0).unsqueeze(1)  # (batch, 1, channel, seq_len)
        conv_outputs = []
        for conv in self.convs:
            c_out = conv(src)
            c_out = torch.sigmoid(c_out)
            b, oc, ck, le = c_out.shape
            c_out = c_out.view(b, oc * ck, le)
            conv_outputs.append(c_out)
        x_cat = torch.cat(conv_outputs, dim=1)    
        x_cat = x_cat.permute(0, 2, 1)            # (batch, length, sum(oc*ck))
        enc_input = self.temp_fc(x_cat)           # (batch, length, d_model)

        # -------------- 位置编码 --------------
        enc_input = enc_input.permute(1, 0, 2)    # (length, batch, d_model)
        enc_input = self.pos_encoder(enc_input)   # (batch, length, d_model)

        # -------------- 手动调用每一层的 TransformerEncoderLayer --------------
        attention_matrices = []
        out = enc_input.permute(1, 0, 2) # (batch, length, d_model)
        for layer in self.transformer_encoder.layers:
            # 1) Multihead Self-Attention
            # 这里将 need_weights=True 来获取注意力矩阵
            src2, attn_weights = layer.self_attn(
                out, 
                out, 
                out,
                attn_mask=src_mask,
                need_weights=True,               # 关键点
                average_attn_weights=False       # 不对多头取均值，保留多头信息
            )
            # 2) Add & Norm
            out = layer.norm1(out + layer.dropout1(src2))
            # 3) FFN
            src2 = layer.linear2(
                layer.dropout(
                    layer.activation(
                        layer.linear1(out)
                    )
                )
            )
            # 4) Add & Norm
            out = layer.norm2(out + layer.dropout2(src2))

            # 记录本层注意力权重
            # attn_weights: [batch_size, n_head, seq_len, seq_len]
            attention_matrices.append(attn_weights)

        return attention_matrices
        
class Encoder(nn.Module):

    def __init__(self, config, device, neuron_num: int):
        super(Encoder, self).__init__()
        self.model_type = 'Dynamic_Encoder'
        self.device = device
        self.d_model = config.MODEL.EMBED_DIM 
        self.neuron_num = neuron_num

        d_model = config.MODEL.EMBED_DIM
        dropout = config.MODEL.DROPOUT
        n_head = config.MODEL.NUM_HEADS
        d_hid = config.MODEL.HIDDEN_SIZE
        n_layers_encoder = config.MODEL.N_LAYERS_ENCODER
        latent_dim = config.MODEL.LATENT_DIM

        # Preprocess
        self.embedding = nn.Linear(neuron_num, d_model, bias=False)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=n_head, 
                                                   dim_feedforward = d_hid,
                                                    dropout = dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers_encoder)

        # Latent space
        self.fc_mu = nn.Linear(d_model, latent_dim)

        self.fc_var = nn.Linear(d_model, latent_dim)
        # self.bn = nn.BatchNorm1d(latent_dim)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Union[Tuple[Tensor, Tensor], Tensor]:
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)
        
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2) # 
        encoded_src = self.transformer_encoder(src, src_mask)
        encoded_src = encoded_src.permute(1, 0, 2) 
        mu = self.fc_mu(encoded_src)
        # 应用BatchNorm1d
        # mu = self.bn(mu.permute(0, 2, 1)).permute(0, 2, 1)

        log_var = self.fc_var(encoded_src)


        return mu, log_var
    
    def get_attention_matrix(self, src: Tensor, src_mask: Tensor = None) -> List[Tensor]:
        """
        新增函数：在不破坏原 forward 的情况下，
        手动逐层调用 TransformerEncoderLayer，收集多头注意力权重。
        
        返回一个列表，其中每个元素对应一层的注意力权重：
        attention_matrices[layer_idx] 的形状是 [batch_size, n_head, seq_len, seq_len]
        """
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)
        
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        out = src.permute(1, 0, 2) # 
        # -------------- 手动调用每一层的 TransformerEncoderLayer --------------
        attention_matrices = []
        for layer in self.transformer_encoder.layers:
            # 1) Multihead Self-Attention
            # 这里将 need_weights=True 来获取注意力矩阵
            src2, attn_weights = layer.self_attn(
                out, 
                out, 
                out,
                attn_mask=src_mask,
                need_weights=True,               # 关键点
                average_attn_weights=False       # 不对多头取均值，保留多头信息
            )
            # 2) Add & Norm
            out = layer.norm1(out + layer.dropout1(src2))
            # 3) FFN
            src2 = layer.linear2(
                layer.dropout(
                    layer.activation(
                        layer.linear1(out)
                    )
                )
            )
            # 4) Add & Norm
            out = layer.norm2(out + layer.dropout2(src2))

            # 记录本层注意力权重
            # attn_weights: [batch_size, n_head, seq_len, seq_len]
            attention_matrices.append(attn_weights)
        return attention_matrices
    
class Decoder(nn.Module):

    def __init__(self, config, device, neuron_num: int, pos=True):
        super(Decoder, self).__init__()
        self.model_type = 'Dynamic_Decoder'
        self.device = device
        self.d_model = config.MODEL.EMBED_DIM
        self.neuron_num = neuron_num
        self.pos = pos

        latent_dim = config.MODEL.LATENT_DIM # dz
        d_model = config.MODEL.EMBED_DIM # d
        n_head = config.MODEL.NUM_HEADS
        d_hid = config.MODEL.HIDDEN_SIZE
        dropout = config.MODEL.DROPOUT
        n_layers_decoder = config.MODEL.N_LAYERS_DECODER

        self.fc_z = nn.Linear(latent_dim, d_model) 
        if self.pos:
            self.pos_encoder = PositionalEncoding(d_model, dropout) # embeding + positional encoding
        decoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_hid, dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, n_layers_decoder)
        self.linear = nn.Linear(d_model, neuron_num) # d->N
        #self.outputLayer = nn.Sigmoid()

    def forward(self, z: Tensor, z_mask: Tensor = None) -> Tensor:
        if z_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            z_mask = nn.Transformer.generate_square_subsequent_mask(len(z)).to(self.device)
        z = self.fc_z(z)
        if self.pos:
            z = self.pos_encoder(z)
        z = z.permute(1, 0, 2) # [seq_len, batch_size, embed_dim] -> [batch_size, seq_len, embed_dim]
        decoded_z = self.transformer_decoder(z, z_mask)
        decoded_z = decoded_z.permute(1, 0, 2) # [batch_size, seq_len, embed_dim] -> [seq_len, batch_size, embed_dim]
        #return self.outputLayer(self.linear(decoded_z)) # delete sigmoid
        return self.linear(decoded_z) 

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # so that pe will not be considered as weights

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
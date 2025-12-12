import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from functools import partial
# from mamba_ssm import Mamba2


# import itertools
# import numpy as np
# import pandas as pd
# from einops import rearrange

BN = partial(nn.BatchNorm1d, eps=5e-3, momentum=0.1)
LN = partial(nn.LayerNorm, eps=1e-5)


class FlashAttentionTransformerEncoder(nn.Module):
    '''
    https://gist.github.com/kklemon/98e491ff877c497668c715541f1bf478
    '''
    def __init__(
        self,
        dim_model,
        num_layers,
        num_heads=None,
        dim_feedforward=None,
        dropout=0.0,
        norm_first=False,
        activation=F.gelu,
        rotary_emb_dim=0,
    ):
        super().__init__()

        try:
            from flash_attn.bert_padding import pad_input, unpad_input
            from flash_attn.modules.block import Block
            from flash_attn.modules.mha import MHA
            from flash_attn.modules.mlp import Mlp
        except ImportError:
            raise ImportError('Please install flash_attn from https://github.com/Dao-AILab/flash-attention')
        
        self._pad_input = pad_input
        self._unpad_input = unpad_input

        if num_heads is None:
            num_heads = dim_model // 64
        
        if dim_feedforward is None:
            dim_feedforward = dim_model * 4

        if isinstance(activation, str):
            activation = {
                'relu': F.relu,
                'gelu': F.gelu
            }.get(activation)

            if activation is None:
                raise ValueError(f'Unknown activation {activation}')

        mixer_cls = partial(
            MHA,
            num_heads=num_heads,
            use_flash_attn=True,
            rotary_emb_dim=rotary_emb_dim
        )

        mlp_cls = partial(Mlp, hidden_features=dim_feedforward)
        #mlp_cls = partial(Mlp, hidden_features=dim_feedforward,out_features=1024)

        self.layers = nn.ModuleList([
            Block(
                dim_model,
                mixer_cls=mixer_cls,
                mlp_cls=mlp_cls,
                resid_dropout1=dropout,
                resid_dropout2=dropout,
                prenorm=norm_first,
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x, src_key_padding_mask=None):
        batch, seqlen = x.shape[:2]
   
        if src_key_padding_mask is None:
            for layer in self.layers:
                x = layer(x)
        else:
            x, indices, cu_seqlens, max_seqlen_in_batch = self._unpad_input(x, ~src_key_padding_mask)
        
            for layer in self.layers:
                x = layer(x, mixer_kwargs=dict(
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen_in_batch
                ))

            x = self._pad_input(x, indices, batch, seqlen)
            
        return x


class Tokenizer(nn.Module):
    '''
    reference: https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/ft_transformer.py
    '''
    def __init__(self, d_numerical: int, d_token: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_numerical + 1, d_token))
        self.bias = nn.Parameter(torch.empty(d_numerical, d_token)) if bias else None
        
        
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        x_num = torch.cat(
            [torch.ones(len(x_num), 1, device=x_num.device)] + [x_num],
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        
        
        return x.bfloat16()

class TokenizedFAEncoder(nn.Module):
    def __init__(self, d_numerical, d_token, bias=True, num_layers=7, dropout=0.1, norm_type=None):
        super().__init__()

        self.input_embedding = Tokenizer(d_numerical, d_token, bias)

        self.tx_encoder = FlashAttentionTransformerEncoder(
            dim_model=d_token,
            num_heads=8,
            dim_feedforward=1024,
            dropout=dropout,
            norm_first=False,
            activation=F.gelu,
            rotary_emb_dim=0,
            num_layers=num_layers,
        )

        self.norm_type = norm_type

        if self.norm_type == 'batchnorm':   
            self.norm = BN(d_numerical)
        elif self.norm_type == 'layernorm':
            self.norm = LN(d_numerical)
        else:
            self.norm = None

    def forward(self, x, mask=None):
        

        
        if self.norm_type is not None:
            x = self.norm(x)
        else:
            x = x    
        
        x = self.input_embedding(x)


        if mask is not None:
           
            z = self.tx_encoder(
                x=x,
                src_key_padding_mask=mask==0,
            )

        else:
            z = self.tx_encoder(x=x)
        
        return z



# class TokenizedMamba(nn.Module):
#     def __init__(self, d_numerical, d_token, bias=True, num_layers=7, dropout=0.1, norm_type=None):
#         super().__init__()
#         self.input_embedding = Tokenizer(d_numerical, d_token, bias)

#         self.tx_encoder = nn.ModuleList([Mamba2(d_model=d_token, d_state=16, d_conv=4, expand=2) for _ in range(num_layers)])

#         self.norm_type = norm_type

#         if self.norm_type == 'batchnorm':   
#             self.norm = BN(d_numerical)
#         elif self.norm_type == 'layernorm':
#             self.norm = LN(d_numerical)
#         else:
#             self.norm = None

#     def forward(self, x, mask=None):

#         if self.norm_type is not None:
#             x = self.norm(x)
#         else:
#             x = x

#         x = self.input_embedding(x)


#         for layer in self.tx_encoder:
#             x = layer(x)

        
#         return x
    
class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x 的形状应为 (batch_size, sequence_length, input_dim)
        for attn_layer in self.layers:
            x, _ = attn_layer(x, x, x)
            x = self.dropout(x)
            x = self.layer_norm(x + x)
        return x
    
class MLP(nn.Module):
    def __init__(
        self, 
        input_dim,
        hidden_dim=None,
        output_dim=None,
        dropout=0.1,
        multiple_of=128,
        bias1=True,
        bias2=True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        output_dim = output_dim if output_dim is not None else input_dim
        hidden_dim = (
            hidden_dim if hidden_dim is not None else int(8 * input_dim / 3)
        )

        hidden_dim = (hidden_dim + multiple_of - 1) // multiple_of * multiple_of

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias1, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias2, **factory_kwargs)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
    

# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mlp.py#L99C1-L136C57
class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        bias1=True,
        bias2=True,
        multiple_of=128,
        dropout=0.1,
        return_residual=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias1, **factory_kwargs)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        y = self.fc1(x)
     
        y = F.glu(y, dim=-1)
        y = self.dropout(y)
        y = self.fc2(y)
  
        return y if not self.return_residual else (y, x)

class CrossAttention(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x, y, mask=None):
        # x 和 y 的形状应为 (batch_size, sequence_length, input_dim)
        for attn_layer in self.layers:
            x, _ = attn_layer(x, y, y, key_padding_mask=mask)
            x = self.dropout(x)
            x = self.layer_norm(x)
        return x


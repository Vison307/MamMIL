# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Compared with V1
keeep CLS token at the last positiion for both forward SSM and backward SSM
"""
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
import numpy as np

from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCNConv, GraphConv, GatedGraphConv, GATConv, SGConv, GINConv, GENConv, DeepGCNLayer
from torch_geometric.nn import GraphConv, TopKPooling, SAGPooling
from torch_geometric.nn import global_mean_pool as gavgp, global_max_pool as gmp, global_add_pool as gap
from torch_geometric.transforms.normalize_features import NormalizeFeatures

from einops import rearrange, repeat

from mamba.mamba_ssm import BiMamba2

import math
from timm.models.layers import trunc_normal_, lecun_normal_

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
    

class PPEG(nn.Module):
    def __init__(self, in_channels, out_channels, use_conv3, use_conv5, use_conv7, activation=nn.SiLU()):
        super().__init__()
        self.use_conv3 = use_conv3
        self.use_conv5 = use_conv5
        self.use_conv7 = use_conv7

        if use_conv3:
            self.conv3 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                groups=in_channels,
                padding=1,
            )
        if use_conv5:
            self.conv5 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=5,
                groups=in_channels,
                padding=2,
            )
        if use_conv7:
            self.conv7 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                groups=in_channels,
                padding=3,
            )

    def forward(self, x, cls_token_position=None):
        # x: [B, L, D]
        if cls_token_position:
            cls_token = x[:, cls_token_position, :]
            x_in = torch.cat([x[:, :cls_token_position, :], x[:, cls_token_position+1:, :]], dim=1)
        else:
            x_in = x
        L = x_in.shape[1]

        _H, _W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
        add_length = _H * _W - L
        x_in = torch.cat([x_in, x_in[:,:add_length, :]],dim=1) #[B, H^2, D]
        x_in = rearrange(x_in, "b (h w) d -> b d h w", h=_H, w=_W) #[B, D, H, W]

        out = x_in
        if self.use_conv3:
            out = out + self.conv3(x_in)
        if self.use_conv5:
            out = out + self.conv5(x_in)
        if self.use_conv7:
            out = out + self.conv7(x_in)

        out = rearrange(out, "b d h w -> b (h w) d")[:, :L, :]

        if cls_token_position:
            out = torch.cat([out[:, :cls_token_position, :], cls_token.unsqueeze(1), out[:, cls_token_position:, :]], dim=1)

        return out


class MamMIL2BiMamba(nn.Module):
    def __init__(self, input_dim, n_directions=4, bidirectional=True, embed_dim=192, layer_n=2, d_state=128, d_conv=4, expand=2, act='relu', dropout=0., num_classes=1000, survival=False,  **kwargs):
        super(MamMIL2BiMamba, self).__init__()

        # ----> First MLP to project the input instance features
        self.ft_head = [nn.Linear(input_dim, embed_dim)]
        if act.lower() == 'relu':
            self.ft_head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self.ft_head += [nn.GELU()]
        if dropout:
            self.ft_head += [nn.Dropout(dropout)]
        self.ft_head = nn.Sequential(*self.ft_head)
        
        # Mamba blocks
        self.layers = nn.ModuleList()
        for _ in range(layer_n):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    BiMamba2(
                        d_model=embed_dim,
                        d_state=d_state,  
                        d_conv=d_conv,    
                        expand=expand,
                    ),
                )
            )
        
        # self.ppeg = nn.ModuleList(
        #     [
        #         PPEG(embed_dim, embed_dim, use_conv3=True, use_conv5=True, use_conv7=True)
        #         for _ in range(layer_n)
        #     ]
        # )

        self.ppeg = torch.nn.ModuleList()
        for i in range(1, layer_n):
            conv = GENConv(embed_dim, embed_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(embed_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=True)
            self.ppeg.append(layer)

        # ----> Aggregator
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # self.attention = Attn_Net_Gated(L=embed_dim, D=embed_dim, dropout=dropout, n_classes=1)

        # ----> Classification Head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.head.apply(segm_init_weights)
        self.apply(
            partial(
                _init_weights,
                n_layer=layer_n
            )
        )

        self.survival = survival
        if survival:
            from utils.survival_utils import NLLSurvLoss
            self.loss_fn = NLLSurvLoss(alpha=0)

    def forward(self, **kwargs):
        G = kwargs['data']
        label = kwargs['label']
        Y = label.long()

        x = G.x
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        h = x.float()  # [B, n, 1024]
        # print(h.shape)

        h = self.ft_head(h)

        edge_index = G.edge_index
        unshuffle_index = torch.arange(h.shape[1])
        preorder_index = G.dfs_preorder_index
        postorder_index = G.dfs_postorder_index
        levelorder_index = G.bfs_levelorder_index
        index_list = [unshuffle_index, preorder_index, postorder_index, levelorder_index]
        for idx, (layer, ppeg) in enumerate(zip(self.layers, self.ppeg)):
            h_ = h
            h = layer[0](h) # LN
            h = layer[1](h)
            if idx < len(self.layers) - 1:
                h = ppeg(h, edge_index)
            h = h + h_

        h = self.norm(h)
        A = self.attention(h) # [B, n, K]
        A = torch.transpose(A, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        h = torch.bmm(A, h) # [B, K, 512]
        h = h.squeeze(0)
        # print(h.shape)

        logits = self.head(h) # [B, num_classes] 1xnum_classes

        if self.survival:
            c = kwargs['c']
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            loss = self.loss_fn(hazards=hazards, S=S, Y=Y, c=c)
            
            results_dict = {'logits': logits, 'Y_hat': Y_hat, 'hazards': hazards, 'S': S, 'loss': loss}
            return results_dict
        
        Y_prob = F.softmax(logits, dim=1)  # [K, num_classes] 1xnum_classes
        Y_hat = torch.argmax(Y_prob, dim=1)  # [K] [1]

        loss = F.cross_entropy(logits, Y)

        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'loss': loss, 'bag_features': x}
        return results_dict


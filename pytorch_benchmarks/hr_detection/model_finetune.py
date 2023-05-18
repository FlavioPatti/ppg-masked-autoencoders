# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from json import encoder

import torch
import torch.nn as nn
import numpy as np
#from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import Block
from util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_flexible, get_1d_sincos_pos_embed_from_grid
from util.misc import concat_all_gather
from util.patch_embed import PatchEmbed_new, PatchEmbed_org
from timm.models.swin_transformer import SwinTransformerBlock


class MaskedAutoencoderViT_without_decoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, stride=10, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, typeExp="freq+time",
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 audio_exp=False, alpha=0.0, temperature=.2, mode=0, contextual_depth=8,
                 use_custom_patch=False, split_pos=False, pos_trainable=False, use_nce=False, beta=4.0, decoder_mode=0,
                 mask_t_prob=0.6, mask_f_prob=0.5, mask_2d=False,
                 epoch=0, no_shift=False,
                 ):
        super().__init__()

        self.audio_exp=audio_exp
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        
        self.patch_embed = PatchEmbed_org(img_size, patch_size, in_chans, embed_dim, typeExp)
        self.use_custom_patch = use_custom_patch
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        #self.split_pos = split_pos # not useful
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding

        self.encoder_depth = depth
        self.contextual_depth = contextual_depth
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_norm=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

         # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding

        self.no_shift=no_shift

        self.decoder_mode = decoder_mode
    
        # Transfomer
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_norm=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
      
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2  * in_chans, bias=True) # decoder to patch

        # --------------------------------------------------------------------------
        
        self.norm_pix_loss = norm_pix_loss

        self.patch_size=patch_size
        self.stride=stride

        # audio exps
        self.alpha = alpha
        self.T = temperature
        self.mode = mode
        self.use_nce = use_nce
        self.beta = beta

        self.log_softmax=nn.LogSoftmax(dim=-1)

        self.epoch = epoch

        # Output layer
        self.out_neuron = nn.Linear(in_features=64, out_features=1)
        
        self.initialize_weights()
        
        """
        # Output layer 1
        self.out_neuron_1 = nn.Linear(in_features=256, out_features=128)
        nn.init.constant_(self.out_neuron_1.bias, 0)
        torch.nn.init.xavier_uniform_(self.out_neuron_1.weight)
        # Output layer 2
        self.out_neuron_2 = nn.Linear(in_features=128, out_features=1)
        nn.init.constant_(self.out_neuron_2.bias, 0)
        torch.nn.init.xavier_uniform_(self.out_neuron_2.weight)
        """
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m,nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder_no_mask(self, x, typeExp):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        #x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        contextual_embs=[]
        for n, blk in enumerate(self.blocks):
          x = blk(x)
          """
          if typeExp == "freq+time":
            m = nn.BatchNorm1d(num_features=257, device = 'cuda')
          if typeExp == "time":
            m = nn.BatchNorm1d(num_features=65, device = 'cuda')
          x=m(x)
          m = nn.ReLU()
          x=m(x)
          """
          #if n > self.contextual_depth:
           # contextual_embs.append(self.norm(x))
        if typeExp == "freq+time":
          m = nn.BatchNorm1d(num_features=257, device = 'cuda')
        if typeExp == "time":
          m = nn.BatchNorm1d(num_features=257, device = 'cuda')
        x=m(x)
        m = nn.ReLU()
        x=m(x)
        #contextual_emb = torch.stack(contextual_embs,dim=0).mean(dim=0)
        #print(f"x_max_fin = {x.max()}")
        #print(f"x_min_fin = {x.min()}")
        return x

    def forward(self, imgs, typeExp="freq+time", mask_ratio=0.1):

        print("")
        #print(f"imgs = {imgs.shape}") 
        #(128,4,256,1) for time, (128, 4, 64, 256) for freq+time
        x = self.forward_encoder_no_mask(imgs, typeExp)
        #print(f"x1 = {x.shape}") 
        #(128,257,256) for time, (128,257,256) for freq+time = (N,C,T)
        m = nn.Conv1d(in_channels=257,out_channels=128,kernel_size=4, stride=4, device='cuda')
        x = m(x)
        #print(f"x2= {x.shape}")
        m = nn.Conv1d(in_channels=128,out_channels=64,kernel_size=4, stride=4, device='cuda')
        x=m(x)
        #print(f"x3= {x.shape}")
        m = nn.AvgPool1d(16)
        x = m(x)
        #print(f"x4 = {x.shape}")
        x = x.flatten(1)
       # print(f"x5 = {x.shape}") #(128,64)

        """
        #first istance of regression
        m = nn.Linear(in_features=256, out_features=128, device='cuda')
        x=m(x)
        m = nn.ReLU()
        x=m(x)
        m = nn.BatchNorm1d(num_features=128, device = 'cuda')
        x=m(x)

        #print(f"x4 = {x.shape}") (128,128)

        #second istance of regression
        m = nn.Linear(in_features=128, out_features=64, device = 'cuda')
        x=m(x)
        m = nn.ReLU()
        x=m(x)
        m = nn.BatchNorm1d(num_features=64, device = 'cuda')
        x=m(x)
        """
        
        #output layer
        x = self.out_neuron(x)
        #print(f"x6 = {x.shape}") #(64,1)
         
        #loss = self.forward_loss(imgs, pred, norm_pix_loss=self.norm_pix_loss)
        #pred, _, _ = self.forward_decoder(emb_enc, ids_restore)  # [N, L, p*p*3]
        #loss_contrastive = torch.FloatTensor([0.0]).cuda()
        return x
from math import ceil
from typing import Dict, Any, Optional
import torch.nn as nn
from pytorch_benchmarks.hr_detection.models_mae import MaskedAutoencoderViT
from pytorch_benchmarks.hr_detection.model_finetune import MaskedAutoencoderViT_without_decoder
from functools import partial

def get_reference_model(model_name: str, model_config: Optional[Dict[str, Any]] = None):
    if model_name == 'vit_freq+time_pretrain':
        print(f"ViT Freq+Time Pretrain")
        return MaskedAutoencoderViT(
        img_size = (64,256), in_chans = 4, mask_2d=True, typeExp = "freq+time",
        patch_size=8, embed_dim=256, depth=12, num_heads=16,
        decoder_embed_dim=256, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6) )
        
        #depth = 4,8,12
        #heads = 4,8,16
        #embed = 64,128,256
        
    if model_name == 'vit_time_pretrain':
        print(f"ViT Time Pretrain")
        return MaskedAutoencoderViT(
        img_size = 256, in_chans = 4, mask_2d=False, typeExp = "time",
        patch_size=1, embed_dim=256, depth=12, num_heads=16,
        decoder_embed_dim=256, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6) )
    if model_name == 'vit_freq+time_finetune':
        print(f"ViT Freq+Time Finetune")
        return MaskedAutoencoderViT_without_decoder(
        img_size = (64,256), in_chans = 4, mask_2d=True, typeExp = "freq+time",
        patch_size=8, embed_dim=256, depth=12, num_heads=16,
        decoder_embed_dim=256, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6) )
    if model_name == 'vit_time_finetune':
        print(f"ViT Time Finetune")
        return MaskedAutoencoderViT_without_decoder(
        img_size = 256, in_chans = 4, mask_2d=False, typeExp = "time",
        patch_size=1, embed_dim=64, depth=4, num_heads=4,
        decoder_embed_dim=64, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6) )
            
    else:
        raise ValueError(f"Unsupported model name {model_name}")
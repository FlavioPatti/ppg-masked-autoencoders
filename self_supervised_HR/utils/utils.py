import pathlib
import random
import numpy as np
import pdb
import copy
import pickle
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm.optim.optim_factory as optim_factory
import torch.nn as nn
from self_supervised_HR.time.model_pretrain import MaskedAutoencoderViT_time
from self_supervised_HR.time.model_finetune import MaskedAutoencoderViT_without_decoder_time
from self_supervised_HR.frequency.model_pretrain import MaskedAutoencoderViT_freq
from self_supervised_HR.frequency.model_finetune import MaskedAutoencoderViT_without_decoder_freq
from functools import partial

def save_checkpoint_pretrain(state, filename="checkpoint_model_pretrain"):
    print("=> Saving pretrained checkpoint")
    torch.save(state,filename)

def load_checkpoint_pretrain(model, checkpoint):
    print("=> Loading pretrained checkpoint")
    model.load_state_dict(checkpoint['state_dict'])

class LogCosh(nn.Module):
    def __init__(self):
      super(LogCosh, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
      x = input - target
      return torch.mean(x + nn.Softplus()(-2*x) - torch.log(torch.tensor(2.)))

#Testing different architectures with:
#depth = 4,8,12
#heads = 4,8,16
#embed = 64,128,256

def get_reference_model(model_name: str):

    if model_name == 'vit_freq_pretrain':
        print(f"=> ViT Freq Pretrain")
        return MaskedAutoencoderViT_freq(
        img_size = (64,256), in_chans = 4, mask_2d=True, type = "freq",
        patch_size=8, embed_dim=64, depth=4, num_heads=16,
        decoder_embed_dim=64, decoder_num_heads=16, 
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6) )

    if model_name == 'vit_time_pretrain':
        print(f"=> ViT Time Pretrain")
        return MaskedAutoencoderViT_time(
        img_size = 256, in_chans = 4, mask_2d=False, type = "time",
        patch_size=1, embed_dim=256, depth=12, num_heads=16,
        decoder_embed_dim=256, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6) )

    if model_name == 'vit_freq_finetune':
        print(f"=>ViT Freq Finetune")
        return MaskedAutoencoderViT_without_decoder_freq(
        img_size = (64,256), in_chans = 4, mask_2d=True, type = "freq",
        patch_size=8, embed_dim=64, depth=4, num_heads=16,
        decoder_embed_dim=64, decoder_num_heads=16, 
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6) )
        
    if model_name == 'vit_time_finetune':
        print(f"=> ViT Time Finetune")
        return MaskedAutoencoderViT_without_decoder_time(
        img_size = 256, in_chans = 4, mask_2d=False, type = "time",
        patch_size=1, embed_dim=256, depth=12, num_heads=16,
        decoder_embed_dim=256, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6) )
            
    else:
        raise ValueError(f"Unsupported model name {model_name}")


def get_default_optimizer(net: nn.Module, task):
    
  if task == "pretrain":
    print(f"=> Loading pretrain optimizer: AdamW")
    #setting optimizer for masked autoencoders
    param_groups = optim_factory.param_groups_weight_decay(net, 0.01) #weight_decay
    return optim.AdamW(param_groups, lr=0.001, betas=(0.9, 0.95))
  if task == "finetune":
    print(f"=> Loading finetune optimizer: Adam")
    #setting optimizer for hr estimation
    return optim.Adam(net.parameters(), lr=0.001)


def get_default_criterion(task):
    if task == "pretrain":
      print(f"=> Loading pretrain criterion: MSE Loss")
    #setting criterion for masked autoencoders
      return nn.MSELoss()
    if task == "finetune":
      print(f"=> Loading finetune criterion: LogCosh Loss")
    #setting criterion for hr estimation
      return LogCosh()


"""plot audio from samples"""
def plot_audio(x, type, num_sample, epoch):
  plt.figure(figsize=(15, 5))
  time = np.linspace(0, 8, num=256)
  plt.xlim([0,8])
  plt.plot(time, x)
  plt.ylabel('Signal wave')
  plt.xlabel('Time (s)')
  plt.title(f"Heatmap PPG: sample {num_sample}")  
  plt.savefig(f'./self_supervised_HR/imgs/{type}/audio{num_sample}_epoch{epoch}.png') 
  
"""plot heart rates"""
def plot_heart_rates(pred, target, type, epoch):
  plt.figure(figsize=(15, 5))
  plt.plot(pred, label='Predictions')
  plt.plot(target, label='True Target')
  plt.xlabel('Time (s)')
  plt.ylabel('Heart Rate (BPM)')
  plt.title('Predictions vs Targets')
  plt.legend()
  plt.savefig(f'./self_supervised_HR/imgs/{type}/HR_epoch{epoch}.png') 
  
"""plot heatmaps from samples"""
def plot_heatmap(x, type, num_sample, epoch):
  _, ax = plt.subplots()
  left = 0
  right= 8
  bottom = 4
  top = 0 
  extent = [left,right, bottom, top]
  im = ax.imshow(x, cmap = 'hot', interpolation = 'hanning', extent = extent)
  ax.figure.colorbar(im, ax = ax)
  ax.set_title(f"Heatmap PPG: sample {num_sample}")  
  plt.xlabel('Time (s)')
  plt.ylabel('Frequency (Hz)')
  plt.savefig(f'./self_supervised_HR/imgs/{type}/specto{num_sample}_epoch{epoch}.png') 

def unpatchify(imgs, type):
        """
        x: (N, L, patch_size**2 *4)
        specs: (N, 4, H, W)
        """ 
        if type == "freq":
         p = 8 
         h = 64//p
         w = 256//p
        elif type == "time":
          p = 1
          h = 256//p
          w = 1//p

        x = imgs.reshape(shape=(imgs.shape[0], h, w, p, p, 4))
        x = torch.einsum('nhwpqc->nchpwq', x)
        specs = x.reshape(shape=(x.shape[0], 4, h * p, w * p))

        return specs


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, fmt='f', name='meter'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get(self):
        return float(self.avg)

    def __str__(self):
        fmtstr = '{:' + self.fmt + '}'
        return fmtstr.format(float(self.avg))


class EarlyStopping():
    """
    stop the training when the loss does not improve.
    """
    def __init__(self, patience=20, mode='min'):
        if mode not in ['min', 'max']:
            raise ValueError("Early-stopping mode not supported")
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_val = None

    def __call__(self, val):
        val = float(val)
        if self.best_val is None:
            self.best_val = val
        elif self.mode == 'min' and val < self.best_val:
            self.best_val = val
            self.counter = 0
        elif self.mode == 'max' and val > self.best_val:
            self.best_val = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early Stopping!")
                return True
        return False


def seed_all(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    return seed

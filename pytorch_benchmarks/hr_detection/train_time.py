from typing import Dict
from timm.models.layers import pad_same
from torch.nn.modules import normalization
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_benchmarks.utils import AverageMeter
import math
import sys
from typing import Iterable
import torch
import util.misc as misc
import util.lr_sched as lr_sched
import timm.optim.optim_factory as optim_factory
import torchaudio
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import librosa

Z_NORM = True
MIN_MAX_NORM = False
PLOT_HEATMAP = True
BATCH_NORM = False
GROUP_NORM = False

def unpatchify(x):
        """
        x: (N, L, patch_size**2 *3)
        specs: (N, 1, H, W)
        """
       # p = self.patch_embed.patch_size[0]    
        p = 1
        h = 256//p
        w = 1//p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 4))
        x = torch.einsum('nhwpqc->nchpwq', x)
        specs = x.reshape(shape=(x.shape[0], 4, h * p, w * p))
        return specs

"""plot heatmap"""
def plot_audio(x, typeExp, num_sample, epoch):
  plt.figure(figsize=(15, 5))
  time = np.linspace(0, 8, num=256)
  plt.xlim([0,8])
  plt.plot(time, x)
  plt.ylabel(' signal wave')
  plt.xlabel('time (s)')
  plt.title(f"Heatmap PPG: sample {num_sample}")  
  plt.savefig(f'./Benchmark_hr_detection/pytorch_benchmarks/imgs/{typeExp}/audio{num_sample}_epoch{epoch}.png') 


class LogCosh(nn.Module):
    def __init__(self):
        super(LogCosh, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = input - target
        return torch.mean(x + nn.Softplus()(-2*x) - torch.log(torch.tensor(2.)))


def get_default_optimizer(net: nn.Module, task):

  if task == "pretrain":
    #setting optimizer for masked autoencoders
    param_groups = optim_factory.param_groups_weight_decay(net, 0.01) #weight_decay
    return optim.AdamW(param_groups, lr=0.001, betas=(0.9, 0.95))
  if task == "finetune":
    #setting optimizer for hr estimation
    return optim.Adam(net.parameters(), lr=0.001)


def get_default_criterion(task):
    if task == "pretrain":
    #setting criterion for masked autoencoders
        return nn.MSELoss()
    else:
    #setting criterion for hr estimation
        return LogCosh()


def _run_model(model, sample, target, criterion):
    output = model(sample)
    #print(f"output = {output.shape}")
    #print(f"target = {target.shape}")
    loss = criterion(output, target)
    return output, loss


def train_one_epoch_masked_autoencoder_time(model: torch.nn.Module,
                    data_loader: DataLoader, criterion: torch.nn.MSELoss, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    accum_iter = 1
    accum_iter2 = 366

    optimizer.zero_grad()

    # set model epoch
    model.epoch = epoch

    for data_iter_step, (samples, _labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        PLOT_HEATMAP = False
        #print(f"data_iter_step = {data_iter_step}")
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            #print(f"adjust_learning_rate")
            #print(f"epoch = {data_iter_step / len(data_loader) + epoch}")
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            #print(f"optimizer = {optimizer}")

        if data_iter_step == accum_iter2:
          PLOT_HEATMAP = True
        
        #COME NORMALIZZAZIONE E' GIUSTA?
        if Z_NORM:
          mean = samples[:,0,:].mean()
          std = samples[:,0,:].std()
          samples[:,0,:] = (samples[:,0,:]-mean) / std

        samples = torch.tensor(np.expand_dims(samples, axis= -1))

        #print(f"shape = {samples.shape}")
        #print(f" max channel 0 = {samples[:,0,:,:].max()}")
        #print(f" min channel 0 = {samples[:,0,:,:].min()}")
        
        #Normalize values into range [0,1] to avoid NaN loss
        if MIN_MAX_NORM:
          channel_1 = samples[:,0,:,:]
          for i in range(samples.shape[0]):
            ch1 = channel_1[i].numpy()
            max_ch1 = np.max(ch1)
            min_ch1 = np.min(ch1)
            #print(f"max = {max_ch1}")
            #print(f"min = {min_ch1}")
            ch1 = (ch1 - min_ch1) / (max_ch1-min_ch1)

            max_ch1 = np.max(ch1)
            min_ch1 = np.min(ch1)
            #print(f"max = {max_ch1}")
            #print(f"min = {min_ch1}")
            samples[i,0,:,:] = torch.tensor(ch1, dtype = float)

        if PLOT_HEATMAP:
          #print("entro")
          for idx in range(50,51):
            sample = samples[idx,:,:,:]
            ch1 = sample[0].detach().numpy() 
            plot_audio(x= ch1, typeExp = "input",num_sample = idx, epoch=epoch)
            #print(f"specto {idx} creato")
    
        samples = samples.to(device, non_blocking=True)

        #print(f"specto_samples = {specto_samples.shape}") #[128,4,256,256] = [batch,channels,freq, time]
        
        #print details of the model 
        #print(model)

        with torch.cuda.amp.autocast():
            #loss_a, _, _, _ = _run_model(specto_samples, mask_ratio=0.1)
            loss_a, pred, _, x_masked = model(samples, "time",mask_ratio=0.1)

        signal_rec = unpatchify(pred)
        signal_rec = np.squeeze(signal_rec.to('cpu').detach())
        #print(f"signal shape = {signal_rec.shape}")
            
        if PLOT_HEATMAP:
          for idx in range(50,51):
            masked = signal_rec[idx,0,:].to('cpu')
            masked = masked.detach().numpy()
            plot_audio(x= masked, typeExp = "rec", num_sample = idx, epoch = epoch)
                    
        loss_value = loss_a.item()
        loss_total = loss_a

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_total = loss_total / accum_iter
        loss_scaler(loss_total, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value) #calculate the average of the loss on all the processes of a group

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
    
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    #print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_hr_detection_time(
        epoch: int,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train: DataLoader,
        val: DataLoader,
        device: torch.device):
    model.train()
    avgmae = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    step = 0
    with tqdm(total=len(train), unit="batch") as tepoch:
      #tepoch.set_description(f"Epoch {epoch+1}")
      for sample, target in train:

        if Z_NORM:
          mean = sample[:,0,:].mean()
          std = sample[:,0,:].std()
          sample[:,0,:] = (sample[:,0,:]-mean) / std
        #print(f" mean = {samples[:,0,:].mean()}")
        #print(f"std = {samples[:,0,:].std()}")
        #print(f" max1 = {samples[:,0,:].max()}")
        #print(f" min1 = {samples[:,0,:].min()}")

        sample = torch.tensor(np.expand_dims(sample, axis= -1))
       # print(f"samples shape = {samples.shape}")
        #print(f"specto shape = {specto_samples.shape}")
        
       #Normalize values into range [0,1] to avoid NaN loss
        if MIN_MAX_NORM:
          channel_1 = sample[:,0,:,:]
          for i in range(sample.shape[0]):
            ch1 = channel_1[i].numpy()
            max_ch1 = np.max(ch1)
            min_ch1 = np.min(ch1)
            #print(f"max = {max_ch1}")
            #print(f"min = {min_ch1}")
            ch1 = (ch1 - min_ch1) / (max_ch1-min_ch1)

            max_ch1 = np.max(ch1)
            min_ch1 = np.min(ch1)
            #print(f"max = {max_ch1}")
            #print(f"min = {min_ch1}")
            sample[i,0,:,:] = torch.tensor(ch1, dtype = float)
        
        step += 1
        tepoch.update(1)
        sample, target = sample.to(device), target.to(device)
        output = model(sample, typeExp= "time")
        loss = criterion(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mae_val = F.l1_loss(output, target) # Mean absolute error for hr detection
        avgmae.update(mae_val, sample.size(0))
        avgloss.update(loss, sample.size(0))
        #if step % 100 == 99:
          #tepoch.set_postfix({'loss': avgloss, 'MAE': avgmae})
      val_metrics = evaluate_time(model, criterion, val, device)
      val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
      final_metrics = {
          'loss': avgloss.get(),
          'MAE': avgmae.get(),
      }
      final_metrics.update(val_metrics)
      #tepoch.set_postfix(final_metrics)
      tepoch.close()
    return final_metrics

def evaluate_time(
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device):
    model.eval()
    avgmae = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
        for sample, target in data:

          if Z_NORM:
            mean = sample[:,0,:].mean()
            std = sample[:,0,:].std()
            sample[:,0,:] = (sample[:,0,:]-mean) / std
            #print(f" mean = {samples[:,0,:].mean()}")
            #print(f"std = {samples[:,0,:].std()}")
            #print(f" max1 = {samples[:,0,:].max()}")
            #print(f" min1 = {samples[:,0,:].min()}")
          sample = torch.tensor(np.expand_dims(sample, axis= -1))
       # print(f"samples shape = {samples.shape}")
        #print(f"specto shape = {specto_samples.shape}")
          
       #Normalize values into range [0,1] to avoid NaN loss
          if MIN_MAX_NORM:
            channel_1 = sample[:,0,:,:]
            for i in range(sample.shape[0]):
              ch1 = channel_1[i].numpy()
              max_ch1 = np.max(ch1)
              min_ch1 = np.min(ch1)
              #print(f"max = {max_ch1}")
              #print(f"min = {min_ch1}")
              ch1 = (ch1 - min_ch1) / (max_ch1-min_ch1)

              max_ch1 = np.max(ch1)
              min_ch1 = np.min(ch1)
              #print(f"max = {max_ch1}")
              #print(f"min = {min_ch1}")
              sample[i,0,:,:] = torch.tensor(ch1, dtype = float)

          step += 1
          sample, target = sample.to(device), target.to(device)
          output = model(sample, typeExp= "time")
          loss = criterion(output,target)
          mae_val = F.l1_loss(output, target)
          avgmae.update(mae_val, sample.size(0))
          avgloss.update(loss, sample.size(0))
        final_metrics = {
          'loss': avgloss.get(),
          'MAE': avgmae.get(),
        }
    return final_metrics
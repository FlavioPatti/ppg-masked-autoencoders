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

RESCALE = True
NORMALIZATION = True
PLOT_HEATMAP = False

"""spectogram trasformation and relative parameters"""
sample_rate= 32
n_fft = 510 #freq = nfft/2 + 1 = 256 => risoluzione/granularità dello spettrogramma
win_length = 32
hop_length = 1 # window length = time instants
n_mels = 64 #definisce la dimensione della frequenza di uscita
f_min = 0
f_max = 4

spectrogram_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    normalized=True,
    f_min = f_min,
    f_max = f_max,
    n_mels = n_mels
)

"""plot heatmap"""
def plot_heatmap_spectogram(x, num_sample):
  fig, ax = plt.subplots()
  left = 0
  right= 8
  bottom = 4
  top = 0 
  extent = [left,right, bottom, top]
  im = ax.imshow(x, cmap = 'hot', interpolation = 'hanning', extent = extent)
  cbar = ax.figure.colorbar(im, ax = ax)
  ax.set_title(f"Heatmap PPG: sample {num_sample}")  
  plt.xlabel('Time (s)')
  plt.ylabel('Frequency (Hz)')
  plt.savefig(f'./pytorch_benchmarks/imgs/specto{num_sample}.png') 


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
  else:
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



def train_one_epoch_masked_autoencoder_freq_time(model: torch.nn.Module,
                    data_loader: DataLoader, criterion: torch.nn.MSELoss, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    accum_iter = 10

    optimizer.zero_grad()

    # set model epoch
    model.epoch = epoch

    for data_iter_step, (samples, _labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        #print(f"data_iter_step = {data_iter_step}")
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
            #print(f"optimizer = {optimizer}")
            
        #samples = samples.to(device, non_blocking=True)
        #samples = [128,4,256] = [batch,channel, time]
        #print(f"sample 0 = {samples[0].shape}") #[4,256]
        
        #print(f"samples shape = {samples.shape}")
        specto_samples = torch.narrow(spectrogram_transform(samples), dim=3, start=0, length=256) 
        #print(f"specto shape = {specto_samples.shape}")
        
        #Rescale samples
        if RESCALE:
          specto_samples = np.log10(specto_samples, where=specto_samples>0)

        if NORMALIZATION:
          channel_1 = specto_samples[:,0,:,:]
          for i in range(specto_samples[0]):
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
            specto_samples[i,0,:,:] = torch.tensor(ch1, dtype = float)


          
        if PLOT_HEATMAP:
          sample = specto_samples[80,:,:,:]
          print(f"sample = {sample.shape}")
          label = _labels[80]
          ch1 = sample[0].numpy()
          max_ch1 = np.max(ch1)
          min_ch1 = np.min(ch1)
          print(f"max ch1 = {max_ch1}")
          print(f"min ch1 = {min_ch1}")
          print(f"label = {label} BPM = {float(label/60)} Hz")
          plot_heatmap_spectogram(x= ch1, num_sample = 80)
      

        # comment out when not debugging
        # from fvcore.nn import FlopCountAnalysis, parameter_count_table
        # if data_iter_step == 1:
        #     flops = FlopCountAnalysis(model, samples)
        #     print(flops.total())
        #     print(parameter_count_table(model))
        specto_samples = specto_samples.to(device, non_blocking=True)

        #print(f"specto_samples = {specto_samples.shape}") #[128,4,256,256] = [batch,channels,freq, time]
        
        #print details of the model 
        #print(model)

        with torch.cuda.amp.autocast():
            #loss_a, _, _, _ = _run_model(specto_samples, mask_ratio=0.1)
            loss_a, _, _, _ = model(specto_samples, mask_ratio=0.1)
        #print(f"loss = {loss_a}")
        loss_value = loss_a.item()
        loss_total = loss_a

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        #loss /= accum_iter
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
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_hr_detection(
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
      tepoch.set_description(f"Epoch {epoch+1}")
      for sample, target in train:

        print(f"samples shape = {sample.shape}")
        specto_samples = spectrogram_transform(sample)
        print(f"specto shape = {specto_samples.shape}")
        
        #Rescale samples
        if RESCALE:
          specto_samples = np.log10(specto_samples)

        #Normalize values into range [0,1] to avoid NaN loss
        if NORMALIZATION:
          specto_samples = (specto_samples - min_ch1)/ (max_ch1 - min_ch1)

        if PLOT_HEATMAP:
          sample = sample[80,:,:]
          label = target[80]
          ch1 = sample[0].numpy()
          max_ch1 = np.max(ch1)
          min_ch1 = np.min(ch1)
          print(f"sample = {sample.shape}")
          print(f"max ch1 = {max_ch1}")
          print(f"min ch1 = {min_ch1}")
          print(f"label = {label} BPM = {float(label/60)} Hz")
          plot_heatmap_spectogram(x= ch1, num_sample = 80)

          
        step += 1
        tepoch.update(1)
        sample, target = sample.to(device), target.to(device)
        output, loss = _run_model(model, sample, target, criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mae_val = F.l1_loss(output, target) # Mean absolute error for hr detection
        avgmae.update(mae_val, sample.size(0))
        avgloss.update(loss, sample.size(0))
        if step % 100 == 99:
          tepoch.set_postfix({'loss': avgloss, 'MAE': avgmae})
      val_metrics = evaluate(model, criterion, val, device)
      val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
      final_metrics = {
          'loss': avgloss.get(),
          'MAE': avgmae.get(),
      }
      final_metrics.update(val_metrics)
      tepoch.set_postfix(final_metrics)
      tepoch.close()
    return final_metrics


def evaluate(
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
          print(f"samples shape = {sample.shape}")
          specto_samples = spectrogram_transform(sample)
          print(f"specto shape = {specto_samples.shape}")
          
          #Rescale samples
          if RESCALE:
            specto_samples = np.log10(specto_samples)

          #Normalize values into range [0,1] to avoid NaN loss
          if NORMALIZATION:
            specto_samples = (specto_samples - min_ch1)/ (max_ch1 - min_ch1)

          if PLOT_HEATMAP:
            sample = sample[80,:,:]
            label = target[80]
            ch1 = sample[0].numpy()
            max_ch1 = np.max(ch1)
            min_ch1 = np.min(ch1)
            print(f"sample = {sample.shape}")
            print(f"max ch1 = {max_ch1}")
            print(f"min ch1 = {min_ch1}")
            print(f"label = {label} BPM = {float(label/60)} Hz")
            plot_heatmap_spectogram(x= ch1, num_sample = 80)

          step += 1
          sample, target = sample.to(device), target.to(device)
          output, loss = _run_model(model, sample, target, criterion)
          mae_val = F.mse_loss(output, target)
          avgmae.update(mae_val, sample.size(0))
          avgloss.update(loss, sample.size(0))
        final_metrics = {
          'loss': avgloss.get(),
          'MAE': avgmae.get(),
        }
    return final_metrics

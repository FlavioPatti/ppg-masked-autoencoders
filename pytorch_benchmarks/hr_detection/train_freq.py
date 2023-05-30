from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_benchmarks.utils import AverageMeter
import math
import sys
import torch
import util.misc as misc
import util.lr_sched as lr_sched
import timm.optim.optim_factory as optim_factory
import torchaudio
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pytorch_benchmarks.hr_detection.models_mae import unpatchify_freq

NORMALIZATION = True
PLOT_HEATMAP = False


"""spectogram trasformation and relative parameters"""
sample_rate= 32
n_fft = 510 #freq = nfft/2 + 1 = 256 => risoluzione/granularitÃ  dello spettrogramma
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
def plot_heatmap_spectogram(x, typeExp, num_sample, epoch = 0):
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
  plt.savefig(f'./Benchmark_hr_detection/pytorch_benchmarks/imgs/{typeExp}/specto{num_sample}_epoch{epoch}.png') 


class LogCosh(nn.Module):
    def __init__(self):
      super(LogCosh, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
      x = input - target
      return torch.mean(x + nn.Softplus()(-2*x) - torch.log(torch.tensor(2.)))


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


def _run_model(model, sample, target, criterion):
    output = model(sample)
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
    optimizer.zero_grad()
    # set model epoch
    model.epoch = epoch
  
    for data_iter_step, (samples, _labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        specto_samples = torch.narrow(spectrogram_transform(samples), dim=3, start=0, length=256) 

        if NORMALIZATION:
          specto_samples = np.log10(specto_samples)  
    
        specto_samples = specto_samples.to(device, non_blocking=True)
        loss_a, pred, target, x_masked = model(specto_samples, "freq+time", mask_ratio = 0.1)
        signal_reconstructed = unpatchify_freq(pred)
        if PLOT_HEATMAP:
              plot_heatmap_training(spectro_samples, epoch,)
        loss_value = loss_a
        loss_scaler(loss_value, optimizer, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
    # gather the stats from all processes
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_hr_detection_freq_time(
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

        specto_samples = torch.narrow(spectrogram_transform(sample), dim=3, start=0, length=256) 
        
        if NORMALIZATION:
          specto_samples = np.log10(specto_samples)
                  
        step += 1
        #tepoch.update(1)
        sample, target = specto_samples.to(device), target.to(device)
        
        output, loss = _run_model(model, sample, target, criterion)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mae_val = F.l1_loss(output, target) # Mean absolute error 
        avgmae.update(mae_val, sample.size(0))
        avgloss.update(loss, sample.size(0))
        if step % 100 == 99:
          tepoch.set_postfix({'loss': avgloss, 'MAE': avgmae})
      val_metrics = evaluate_freq_time(model, criterion, val, device)
      val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
      final_metrics = {
          'loss': avgloss.get(),
          'MAE': avgmae.get(),
      }
      final_metrics.update(val_metrics)
      tepoch.set_postfix(final_metrics)
      tepoch.close()
    return final_metrics


def evaluate_freq_time(
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
         
          specto_samples = torch.narrow(spectrogram_transform(sample), dim=3, start=0, length=256) 
          
          if NORMALIZATION:
             specto_samples = np.log10(specto_samples) 
                        
          step += 1
          sample, target = specto_samples.to(device), target.to(device)
          output, loss = _run_model(model, sample, target, criterion)
          mae_val = F.l1_loss(output, target)
          avgmae.update(mae_val, sample.size(0))
          avgloss.update(loss, sample.size(0))
        final_metrics = {
          'loss': avgloss.get(),
          'MAE': avgmae.get(),
        }
    return final_metrics
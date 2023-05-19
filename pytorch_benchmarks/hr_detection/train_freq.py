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

RESCALE = False
Z_NORM = False
MIN_MAX_NORM = True
PLOT_HEATMAP = True


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
def plot_heatmap_spectogram(x, typeExp, num_sample, epoch = 0):
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
  plt.savefig(f'./pytorch_benchmarks/imgs/{typeExp}/specto{num_sample}_epoch{epoch}.png') 


class LogCosh(nn.Module):
    def __init__(self):
        super(LogCosh, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = input - target
        return torch.mean(x + nn.Softplus()(-2*x) - torch.log(torch.tensor(2.)))


def get_default_optimizer(net: nn.Module, task):

  if task == "pretrain":
    print(f"Load pretrain optimizer")
    #setting optimizer for masked autoencoders
    param_groups = optim_factory.param_groups_weight_decay(net, 0.01) #weight_decay
    return optim.AdamW(param_groups, lr=0.001, betas=(0.9, 0.95))
  if task == "finetune":
    print(f"Load finetune optimizer")
    #setting optimizer for hr estimation
    return optim.Adam(net.parameters(), lr=0.001)


def get_default_criterion(task):
    if task == "pretrain":
      print(f"Load pretrain criterion")
    #setting criterion for masked autoencoders
      return nn.MSELoss()
    if task == "finetune":
      print(f"Load finetune criterion")
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
    accum_iter = 1
    accum_iter2 = 366
    optimizer.zero_grad()
    # set model epoch
    model.epoch = epoch
  
    for data_iter_step, (samples, _labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        PLOT_HEATMAP = False
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        if data_iter_step == accum_iter2:
          PLOT_HEATMAP = True

      
        specto_samples = torch.narrow(spectrogram_transform(samples), dim=3, start=0, length=256) 

        if MIN_MAX_NORM:
          specto_samples = np.log10(specto_samples)
          #max_v = specto_samples.max()
          #min_v = specto_samples.min()
          #specto_samples = (specto_samples - min_v) / ( max_v - min_v)
          #print(f"max = {specto_samples.max()}")
          #print(f"min = {specto_samples.min()}")   
    
        specto_samples = specto_samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss_a, pred, target, x_masked = model(specto_samples, "freq+time", mask_ratio = 0.1)

        signal_reconstructed = unpatchify_freq(pred)

        if PLOT_HEATMAP:
          for idx in range(50,51):
            sample = specto_samples[idx,:,:,:].to('cpu')
            ch0 = sample[0].detach().numpy()
            plot_heatmap_spectogram(x= ch0, typeExp = "input",num_sample = idx, epoch = epoch)

        if PLOT_HEATMAP:
          for idx in range(50,51):
            sample = signal_reconstructed[idx,:,:,:].to('cpu')
            ch0 = sample[0].detach().numpy()
            plot_heatmap_spectogram(x= ch0, typeExp = "input_reconstructed",num_sample = idx, epoch = epoch)
            
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

        if Z_NORM:
          mean = sample[:,0,:].mean()
          std = sample[:,0,:].std()
          sample[:,0,:] = (sample[:,0,:]-mean) / std

        specto_samples = torch.narrow(spectrogram_transform(sample), dim=3, start=0, length=256) 
        
        if RESCALE:
          specto_samples = np.log10(specto_samples, where=specto_samples!=0)

        #Normalize values into range [0,1] to avoid NaN loss
        if MIN_MAX_NORM:
          channel_0 = specto_samples[:,0,:,:]
          for i in range(specto_samples.shape[0]):
            ch0 = channel_0[i].numpy()
            max_ch0 = np.max(ch0)
            min_ch0 = np.min(ch0)
            ch0 = (ch0 - min_ch0) / (max_ch0-min_ch0)
            specto_samples[i,0,:,:] = torch.tensor(ch0, dtype = float)
        
          channel_1 = specto_samples[:,1,:,:]
          for i in range(specto_samples.shape[0]):
            ch1 = channel_1[i].numpy()
            max_ch1 = np.max(ch1)
            min_ch1 = np.min(ch1)
            ch1 = (ch1 - min_ch1) / (max_ch1-min_ch1)
            specto_samples[i,1,:,:] = torch.tensor(ch1, dtype = float)

          channel_2 = specto_samples[:,2,:,:]
          for i in range(specto_samples.shape[0]):
            ch2 = channel_2[i].numpy()
            max_ch2 = np.max(ch2)
            min_ch2 = np.min(ch2)
            if (max_ch2 - min_ch2 != 0):
              ch2 = (ch2 - min_ch2) / (max_ch2-min_ch2)
            specto_samples[i,2,:,:] = torch.tensor(ch2, dtype = float)

          channel_3 = specto_samples[:,3,:,:]
          for i in range(specto_samples.shape[0]):
            ch3 = channel_3[i].numpy()
            max_ch3 = np.max(ch3)
            min_ch3 = np.min(ch3)
            ch3 = (ch3 - min_ch3) / (max_ch3-min_ch3)
            specto_samples[i,3,:,:] = torch.tensor(ch3, dtype = float)
          
        step += 1
        tepoch.update(1)
        sample, target = specto_samples.to(device), target.to(device)
        output = model(sample)
        loss = criterion(output, target)
        
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

          if Z_NORM:
            mean = sample[:,0,:].mean()
            std = sample[:,0,:].std()
            sample[:,0,:] = (sample[:,0,:]-mean) / std
         
          specto_samples = torch.narrow(spectrogram_transform(sample), dim=3, start=0, length=256) 
          
          if RESCALE:
            specto_samples = np.log10(specto_samples, where=specto_samples!=0)

          #Normalize values into range [0,1] to avoid NaN loss
          if MIN_MAX_NORM:
            channel_0 = specto_samples[:,0,:,:]
            for i in range(specto_samples.shape[0]):
              ch0 = channel_0[i].numpy()
              max_ch0 = np.max(ch0)
              min_ch0 = np.min(ch0)
              ch0 = (ch0 - min_ch0) / (max_ch0-min_ch0)
              specto_samples[i,0,:,:] = torch.tensor(ch0, dtype = float)
          
            channel_1 = specto_samples[:,1,:,:]
            for i in range(specto_samples.shape[0]):
              ch1 = channel_1[i].numpy()
              max_ch1 = np.max(ch1)
              min_ch1 = np.min(ch1)
              ch1 = (ch1 - min_ch1) / (max_ch1-min_ch1)
              specto_samples[i,1,:,:] = torch.tensor(ch1, dtype = float)

            channel_2 = specto_samples[:,2,:,:]
            for i in range(specto_samples.shape[0]):
              ch2 = channel_2[i].numpy()
              max_ch2 = np.max(ch2)
              min_ch2 = np.min(ch2)
              if (max_ch2 - min_ch2 != 0):
                ch2 = (ch2 - min_ch2) / (max_ch2-min_ch2)
              specto_samples[i,2,:,:] = torch.tensor(ch2, dtype = float)

            channel_3 = specto_samples[:,3,:,:]
            for i in range(specto_samples.shape[0]):
              ch3 = channel_3[i].numpy()
              max_ch3 = np.max(ch3)
              min_ch3 = np.min(ch3)
              ch3 = (ch3 - min_ch3) / (max_ch3-min_ch3)
              specto_samples[i,3,:,:] = torch.tensor(ch3, dtype = float)
            
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
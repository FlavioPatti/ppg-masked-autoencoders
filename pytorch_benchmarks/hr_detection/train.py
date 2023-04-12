from typing import Dict
from timm.models.layers import pad_same
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

        sample = samples[80,:,:]
        print(f"sample = {sample.shape}")
        
        sample_rate= 32
        n_fft = 510 #freq = nfft/2 + 1 = 256 
        win_length = [8,16,32,64]
        hop_length = 1 # window length = time instants
        n_mels = 32
        f_min = 0
        f_max = 4

        for wl in win_length:

          spectrogram_transform = torchaudio.transforms.MelSpectrogram(
              sample_rate = sample_rate,
              n_fft=n_fft,
              win_length=wl,
              hop_length=hop_length,
              center=True,
              pad_mode="reflect",
              power=2.0,
              normalized=True,
              f_min = f_min,
              f_max = f_max,
              n_mels = n_mels
          )


          sample1 = spectrogram_transform(sample)
          print(f"specto shape = {sample1.shape}")

          ch1 = sample1[0].numpy()
          print(f"shape ch1 = {ch1.shape}")
          max_ch1 = np.max(ch1)
          print(f"max_ch1 = {max_ch1}")
          min_ch1 = np.min(ch1)
          print(f"min_ch1 = {min_ch1}")

          plt.imshow(ch1, cmap ='hot', interpolation = 'hanning')
          plt.title(f'Heatmap channel 1')
          plt.xlabel('time')
          plt.ylabel('freq')
          plt.savefig(f'./pytorch_benchmarks/imgs/specto1_wl={wl}.png') 

          """
          ch2 = sample1[1].numpy()
          print(f"shape ch2 = {ch2.shape}")
          max_ch2 = np.max(ch2)
          print(f"max_ch2 = {max_ch2}")
          min_ch2 = np.min(ch2)
          print(f"min_ch2 = {min_ch2}")

          plt.imshow(ch2, cmap ='hot', interpolation = 'hanning')
          plt.title(f'Heatmap channel 2')
          plt.xlabel('time')
          plt.ylabel('freq')
          plt.savefig(f'./pytorch_benchmarks/imgs/specto2_wl={wl}.png') 

          ch3 = sample1[2].numpy()
          print(f"shape ch3 = {ch3.shape}")
          max_ch3 = np.max(ch3)
          print(f"max_ch3 = {max_ch3}")
          min_ch3 = np.min(ch3)
          print(f"min_ch3 = {min_ch3}")

          plt.imshow(ch3, cmap ='hot', interpolation = 'hanning')
          plt.title(f'Heatmap channel 3')
          plt.xlabel('time')
          plt.ylabel('freq')
          plt.savefig(f'./pytorch_benchmarks/imgs/specto3_wl={wl}.png') 

          ch4 = sample1[3].numpy()
          print(f"shape ch4 = {ch4.shape}")
          max_ch4 = np.max(ch4)
          print(f"max_ch4 = {max_ch4}")
          min_ch4 = np.min(ch4)
          print(f"min_ch4 = {min_ch4}")

          plt.imshow(ch4, cmap ='hot', interpolation = 'hanning')
          plt.title(f'Heatmap channel 4')
          plt.xlabel('time')
          plt.ylabel('freq')
          plt.savefig(f'./pytorch_benchmarks/imgs/specto4_wl={wl}.png') 
          """
        break
        
        #Normalize values into range [0,1] to avoid NaN loss

        #Method 1

        #max_value = torch.max(specto_samples)
        #specto_samples = specto_samples/ max_value

        #Method 2

        # Get min, max value aming all elements for each column
        specto_min = np.min(specto_samples.numpy(), axis=tuple(range(specto_samples.ndim-1)), keepdims=1)
        #print(f"min = {specto_min}")
        specto_max = np.max(specto_samples.numpy(), axis=tuple(range(specto_samples.ndim-1)), keepdims=1)
        #print(f"max = {specto_max}")
        # Normalize with those min, max values leveraging broadcasting
        specto_samples = (specto_samples - specto_min)/ (specto_max - specto_min)


        #Method 3 => use MinMaxScaler
        """
        scaler = MinMaxScaler(feature_range=(0,1))
        image_1d = ch2.reshape(-1,1)
        scaler.fit(image_1d)
        normalized_image_1d = scaler.transform(image_1d)
        normalized_image = normalized_image_1d.reshape(ch2.shape)
        """

        #print heatmap of the first spectogram
       # input_data = specto_samples[0]
        #print(input_data)
       # heatmap_specto(input_data)

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
            #print(f"samples = {sample.shape}")
            sample = torch.narrow(spectrogram_transform(sample), dim=3, start=0, length=256) 
            #print(f"specto samples = {sample.shape}")
            # Get min, max value aming all elements for each column
            specto_min = np.min(sample.numpy(), axis=tuple(range(sample.ndim-1)), keepdims=1)
            #print(f"min = {specto_min}")
            specto_max = np.max(sample.numpy(), axis=tuple(range(sample.ndim-1)), keepdims=1)
            #print(f"max = {specto_max}")
            # Normalize with those min, max values leveraging broadcasting
            sample = (sample - specto_min)/ (specto_max - specto_min)
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
            #print(f"samples = {sample.shape}")
            sample = torch.narrow(spectrogram_transform(sample), dim=3, start=0, length=256) 
            #print(f"specto samples = {sample.shape}")
            # Get min, max value aming all elements for each column
            specto_min = np.min(sample.numpy(), axis=tuple(range(sample.ndim-1)), keepdims=1)
            #print(f"min = {specto_min}")
            specto_max = np.max(sample.numpy(), axis=tuple(range(sample.ndim-1)), keepdims=1)
            #print(f"max = {specto_max}")
            # Normalize with those min, max values leveraging broadcasting
            sample = (sample - specto_min)/ (specto_max - specto_min)
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

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import self_supervised_HR.utils.utils as utils
import torch
import util.misc as misc
import util.lr_sched as lr_sched
import numpy as np

def train_one_epoch_masked_autoencoder_time(model: torch.nn.Module,
                    data_loader: DataLoader, criterion: torch.nn.MSELoss, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    normalization = False, plot_heatmap = False, sample_to_plot = 50):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    optimizer.zero_grad()
    model.epoch = epoch

    for data_iter_step, (samples, _labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch)

        if normalization:
          min_v = samples.min()
          max_v = samples.max()
          samples = (samples - min_v) / ( max_v - min_v)

        #img shape (4,256) -> (4,256,1)
        samples = torch.tensor(np.expand_dims(samples, axis= -1))
    
        samples = samples.to(device, non_blocking=True)

        loss, prediction, target, x_masked = model(samples, mask_ratio=0.1)

        #recostruction of the signal to the original shape
        signal_reconstructed = np.squeeze(utils.unpatchify(prediction, type = "time"))
            
        if plot_heatmap:
          ppg_signal = samples[sample_to_plot,0,:,:].to('cpu').detach().numpy() #ppg signal is channel 0
          utils.plot_audio(x = ppg_signal, type="input", num_sample = sample_to_plot, epoch = epoch)

          ppg_signal_masked = signal_reconstructed[sample_to_plot,0,:].to('cpu').detach().numpy()
          utils.plot_audio(x = ppg_signal_masked, type="input_reconstructed", num_sample = sample_to_plot, epoch = epoch)
          
        loss_scaler(loss, optimizer, parameters=model.parameters(),update_grad=True)
        optimizer.zero_grad()
        metric_logger.update(loss=loss)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_hr_detection_time(
        epoch: int,model: nn.Module,criterion: nn.Module,optimizer: optim.Optimizer,
        train: DataLoader,val: DataLoader,device: torch.device,
        normalization = False, plot_heatmap = False, sample_to_plot = 50):
    model.train()
    avgmae = utils.AverageMeter('6.2f')
    avgloss = utils.AverageMeter('2.5f')
    step = 0
    with tqdm(total=len(train), unit="batch") as tepoch:
      tepoch.set_description(f"Epoch {epoch+1}")
      for sample, target in train:

        if normalization:
          min_v = sample.min()
          max_v = sample.max()
          sample = (sample - min_v) / ( max_v - min_v)

        #img shape (4,256) -> (4,256,1)
        sample = torch.tensor(np.expand_dims(sample, axis= -1))
        
        step += 1
        #tepoch.update(1)
        sample, target = sample.to(device), target.to(device)
        
        output = model(sample)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mae_val = F.l1_loss(output, target) # Mean absolute error for hr detection
        avgmae.update(mae_val, sample.size(0))
        avgloss.update(loss, sample.size(0))
        if step % 100 == 99:
          tepoch.set_postfix({'loss': avgloss, 'MAE': avgmae})
      val_metrics = evaluate_time(model, criterion, val, device)
      val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
      final_metrics = {
          'loss': avgloss.get(),
          'MAE': avgmae.get(),
      }
      final_metrics.update(val_metrics)
      tepoch.set_postfix(final_metrics)
      tepoch.close()
    return final_metrics

def evaluate_time(
        model: nn.Module,criterion: nn.Module,data: DataLoader,device: torch.device,
        normalization = False,plot_heatmap = False, sample_to_plot = 50):
    model.eval()
    avgmae = utils.AverageMeter('6.2f')
    avgloss = utils.AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
      for sample, target in data:
            
        if normalization:
          min_v = sample.min()
          max_v = sample.max()
          sample = (sample - min_v) / ( max_v - min_v)
          
        #img shape (4,256) -> (4,256,1)
        sample = torch.tensor(np.expand_dims(sample, axis= -1))
        
        step += 1
        sample, target = sample.to(device), target.to(device)
        output = model(sample)
        loss = criterion(output, target)
        mae_val = F.l1_loss(output, target) # Mean absolute error for hr detection
        avgmae.update(mae_val, sample.size(0))
        avgloss.update(loss, sample.size(0))
      final_metrics = {
        'loss': avgloss.get(),
        'MAE': avgmae.get(),
      }
    return final_metrics

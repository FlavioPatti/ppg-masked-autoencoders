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

def train_one_epoch_masked_autoencoder_freq(model: torch.nn.Module,
                    data_loader: DataLoader, criterion: torch.nn.MSELoss, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    plot_heatmap = False, normalization = False, sample_to_plot = 50):
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
          samples = np.log10(samples)
    
        samples = samples.to(device, non_blocking=True)
        
        loss, prediction, target, x_masked = model(samples, mask_ratio = 0.1)
        
        #recostruction of the signal to the original shape
        signal_reconstructed = utils.unpatchify(prediction, type = "freq")
        
        if plot_heatmap and data_iter_step == 365:
          ppg_signal = samples[sample_to_plot,0,:,:].to('cpu').detach().numpy()
          utils.plot_heatmap(x = ppg_signal, type="input", num_sample= sample_to_plot, epoch= epoch)
          
          ppg_signal_masked = signal_reconstructed[sample_to_plot,0,:,:].to('cpu').detach().numpy()
          utils.plot_heatmap(x = ppg_signal_masked, type="input_reconstructed", num_sample = sample_to_plot, epoch = epoch)

        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
        optimizer.zero_grad()
        metric_logger.update(loss=loss)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_hr_detection_freq(
        epoch: int,model: nn.Module,criterion: nn.Module,optimizer: optim.Optimizer,
        train: DataLoader,val: DataLoader,device: torch.device, 
        plot_heart_rate = False, normalization = False):
    model.train()
    avgmae = utils.AverageMeter('6.2f')
    avgloss = utils.AverageMeter('2.5f')
    step = 0
    with tqdm(total=len(train), unit="batch") as tepoch:
      tepoch.set_description(f"Epoch {epoch+1}")
      for sample, target in train:
        
        if normalization:
          sample = np.log10(sample)
                
        step += 1
        #tepoch.update(1)
        sample, target = sample.to(device), target.to(device)
        
        output = model(sample)
        loss = criterion(output, target)
        
        if plot_heart_rate and step == 365:
          pred = output.to('cpu').detach().numpy()
          true_target = target.to('cpu').detach().numpy()
          utils.plot_heart_rates(pred = pred, target = true_target, type="HR", epoch = epoch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mae_val = F.l1_loss(output, target) # Mean absolute error for hr detection
        avgmae.update(mae_val, sample.size(0))
        avgloss.update(loss, sample.size(0))
        if step % 100 == 99:
          tepoch.set_postfix({'loss': avgloss, 'MAE': avgmae})
      val_metrics = evaluate_freq(model, criterion, val, device, normalization = normalization)
      val_metrics = {'val_' + k: v for k, v in val_metrics.items()}
      final_metrics = {
          'loss': avgloss.get(),
          'MAE': avgmae.get(),
      }
      final_metrics.update(val_metrics)
      tepoch.set_postfix(final_metrics)
      tepoch.close()
    return final_metrics

def evaluate_freq(
        model: nn.Module,criterion: nn.Module,data: DataLoader,device: torch.device, normalization = False):
    model.eval()
    avgmae = utils.AverageMeter('6.2f')
    avgloss = utils.AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
        for sample, target in data: 
          
          if normalization:
              sample = np.log10(sample)
              
          step += 1
          sample, target = sample.to(device), target.to(device)
          output = model(sample)
          
          output = post_processing(output)
          output = output.cuda()
          loss = criterion(output, target)
          mae_val = F.l1_loss(output, target) # Mean absolute error for hr detection
          avgmae.update(mae_val, sample.size(0))
          avgloss.update(loss, sample.size(0))
        final_metrics = {
          'loss': avgloss.get(),
          'MAE': avgmae.get(),
        }
    return final_metrics
  
def evaluate_post_processing_freq(
        model: nn.Module,data: DataLoader,device: torch.device, normalization = False):
    model.eval()
    step = 0
    output_list = []
    target_list = []
    
    with torch.no_grad():
      for sample, target in data: 
        
        if normalization:
            sample = np.log10(sample)
            
        step += 1
        sample, target = sample.to(device), target.to(device)
        output = model(sample)
        
        for i in range(sample.size(0)): #128
          output_list.append(output[i])
          target_list.append(target[i])
      
      output_list = post_processing(output_list)
      print(f"output list = {output_list.shape}")
      print(f"target list = {target_list.shape}")
      MAE_post_proc= F.l1_loss(output_list, target_list) # Mean absolute error for hr detection
      print(f"MAE post proc = {MAE_post_proc}")  
    return 
  
def post_processing(x):
  #apply post-processing
  N = 10
  x_post_proc = []
  for i in range(x.shape[0]):
    if i >= N:
      previous_values = x[i - N : i]
      avg = sum(previous_values) / N
      th = avg / N 
      if x[i] > avg + th:
          x[i] = avg + th
      elif x[i] < avg - th:
          x[i] = avg - th
      x_post_proc.append(x[i])
    else:
      x_post_proc.append(x[i])  
  x_post_proc = torch.Tensor(x_post_proc)
  x_post_proc = torch.unsqueeze(x_post_proc, dim=1)
  return x_post_proc
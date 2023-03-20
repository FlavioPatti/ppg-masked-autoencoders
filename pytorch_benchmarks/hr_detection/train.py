from typing import Dict
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
"""
This module must implement the minimum set of information required to implement a training loop.

In particular, the mandatory and standard functions that needs to be implemented are:

* get_default_optimizer, it takes as input the pytorch model returned by get_reference_model and returns the default optimizer for the task.
* get_default_criterion, it takes no inputs and returns the default loss function for the task.
* train_one_epoch, implements one epoch of training and validation for the benchmark. 
  For the validation part it directly calls the evaluate function. 
  It takes as input an integer specifying the current epoch, the model to be trained, the criterion, the optimizer, 
  the train and val dataloaders and finally the device to be used for the training. It returns a dictionary of tracked metrics.
* evaluate, implement an evaluation step of the model. 
  This step can be both of validation or test depending on the specific dataloader provided as input. 
  It takes as input the model, the criterion, the dataloader and the device. It returns a dictionary of tracked metrics.
  
Optionally, the benchmark may defines and implements the get_default_scheduler function which takes as input 
the optimizer and returns a specified learning-rate scheduler.
"""

class LogCosh(nn.Module):
    def __init__(self):
        super(LogCosh, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = input - target
        return torch.mean(x + nn.Softplus()(-2*x) - torch.log(torch.tensor(2.)))


def get_default_optimizer(net: nn.Module) -> optim.Optimizer:
    return optim.Adam(net.parameters(), lr=0.001)


def get_default_criterion() -> nn.Module:
    return LogCosh()


def _run_model(model, sample, target, criterion, device):
    output = model(sample)
    loss = criterion(output, target)
    return output, loss

def train_one_epoch_masked_autoencoder(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # set model epoch
    model.epoch = epoch
    for data_iter_step, (samples, _labels, _vids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)



        #print(samples.shape)# 64x3x224x224 for img, 64x1x512x128 for audio
        samples = samples.to(device, non_blocking=True)
        
        # comment out when not debugging
        # from fvcore.nn import FlopCountAnalysis, parameter_count_table
        # if data_iter_step == 1:
        #     flops = FlopCountAnalysis(model, samples)
        #     print(flops.total())
        #     print(parameter_count_table(model))


        with torch.cuda.amp.autocast():
            loss_a, _, _, _ = model(samples, mask_ratio=args.mask_ratio)
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

        loss_value_reduce = misc.all_reduce_mean(loss_value) #calcola la media della loss su tutti i processi di un gruppo

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




def evaluate(
        model: nn.Module,
        criterion: nn.Module,
        data: DataLoader,
        device: torch.device) -> Dict[str, float]:
    model.eval()
    avgmae = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    step = 0
    with torch.no_grad():
        for sample, target in data:
            step += 1
            sample, target = sample.to(device), target.to(device)
            output, loss = _run_model(model, sample, target, criterion, device)
            mae_val = F.mse_loss(output, target)
            avgmae.update(mae_val, sample.size(0))
            avgloss.update(loss, sample.size(0))
        final_metrics = {
            'loss': avgloss.get(),
            'MAE': avgmae.get(),
        }
    return final_metrics


def train_one_epoch_hr_detection(
        epoch: int,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        train: DataLoader,
        val: DataLoader,
        device: torch.device) -> Dict[str, float]:
    model.train()
    avgmae = AverageMeter('6.2f')
    avgloss = AverageMeter('2.5f')
    step = 0
    with tqdm(total=len(train), unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        for sample, target in train:
            step += 1
            tepoch.update(1)
            sample, target = sample.to(device), target.to(device)
            output, loss = _run_model(model, sample, target, criterion, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mae_val = F.mse_loss(output, target) # => Mean square error per Masked auto encoder,  mae_val = F.l1_loss(output, target)  => Mean absolute error per hr detection
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


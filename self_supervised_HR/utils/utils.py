import pathlib
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm.optim.optim_factory as optim_factory
from math import ceil
import torch.nn as nn
from self_supervised_HR.time.model_pretrain import MaskedAutoencoderViT_time
from self_supervised_HR.time.model_finetune import MaskedAutoencoderViT_without_decoder_time
from self_supervised_HR.freq.model_pretrain import MaskedAutoencoderViT_freq
from self_supervised_HR.freq.model_finetune import MaskedAutoencoderViT_without_decoder_freq
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
# #embed = 64,128,256

def get_reference_model(model_name: str, dataset: str):
    if model_name == 'temponet':
        return TEMPONet()

    if model_name == 'vit_freq_pretrain':
        print(f"=> ViT Freq Pretrain")
        if dataset == "DALIA" or dataset == "WESAD":
          return MaskedAutoencoderViT_freq(
          img_size = (64,256), in_chans = 4, mask_2d=True, type = "freq",
          patch_size=8, embed_dim=64, depth=4, num_heads=16,
          decoder_embed_dim=64, decoder_num_heads=16, dataset = dataset,
          mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6) )
        else: #ieeeppg
          return MaskedAutoencoderViT_freq(
            img_size = (64,256), in_chans = 5, mask_2d=True, type = "freq",
            patch_size=8, embed_dim=64, depth=4, num_heads=16,
            decoder_embed_dim=64, decoder_num_heads=16, dataset = dataset,
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
        if dataset == "DALIA" or dataset == "WESAD":
          return MaskedAutoencoderViT_without_decoder_freq(
          img_size = (64,256), in_chans = 4, mask_2d=True, type = "freq",
          patch_size=8, embed_dim=64, depth=4, num_heads=16,
          decoder_embed_dim=64, decoder_num_heads=16, 
          mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6) )
        else: #ieeeppg
          return MaskedAutoencoderViT_without_decoder_freq(
          img_size = (64,256), in_chans = 5, mask_2d=True, type = "freq",
          patch_size=8, embed_dim=64, depth=4, num_heads=16,
          decoder_embed_dim=64, decoder_num_heads=16, 
          mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6) )

    if model_name == 'vit_time_finetune':
        print(f"=> ViT Time Finetune")
        return MaskedAutoencoderViT_without_decoder_time(
        img_size = 256, in_chans = 4, mask_2d=False, type = "time",
        patch_size=1, embed_dim=256, depth=12, num_heads=16,
        decoder_embed_dim=64, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6) )
            
    else:
        raise ValueError(f"Unsupported model name {model_name}")

class TempConvBlock(nn.Module):
    """
    Temporal Convolutional Block composed of one temporal convolutional layer.
    The block is composed of :
    - Conv1d layer
    - ReLU layer
    - BatchNorm1d layer
    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param dil: Amount of dilation
    :param pad: Amount of padding
    """
    def __init__(self, ch_in, ch_out, k_size, dil, pad):
        super(TempConvBlock, self).__init__()
        self.tcn = nn.Conv1d(in_channels=ch_in, out_channels=ch_out,
                             kernel_size=k_size, dilation=dil,
                             padding=pad)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features=ch_out)

    def forward(self, x):
        x = self.tcn(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class ConvBlock(nn.Module):
    """
    Convolutional Block composed of:
    - Conv1d layer
    - AvgPool1d layer
    - ReLU layer
    - BatchNorm1d layer
    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param s: Amount of stride
    :param pad: Amount of padding
    """
    def __init__(self, ch_in, ch_out, k_size, s, pad, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels=ch_in, out_channels=ch_out,
                              kernel_size=k_size, stride=s,
                              dilation=dilation, padding=pad)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(ch_out)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Regressor(nn.Module):
    """
    Regressor block composed of:
    - Linear layer
    - ReLU layer
    - BatchNorm1d layer
    :param ft_in: Number of input channels
    :param ft_out: Number of output channels
    """
    def __init__(self, ft_in, ft_out):
        super(Regressor, self).__init__()
        self.fc = nn.Linear(in_features=ft_in, out_features=ft_out)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features=ft_out)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class TEMPONet(nn.Module):
    """
    TEMPONet architecture:
    Three repeated instances of TemporalConvBlock and ConvBlock organized as follows:
    - TemporalConvBlock
    - ConvBlock
    Two instances of Regressor followed by a final Linear layer with a single neuron.
    """
    def __init__(self):
        super().__init__()

        # Parameters
        self.input_shape = (4, 256)  # default for PPG-DALIA dataset
        self.dil = [2, 2, 1, 4, 4, 8, 8]
        self.rf = [5, 5, 5, 9, 9, 17, 17]
        self.ch = [32, 32, 64, 64, 64, 128, 128, 128, 128, 256, 128]

        # 1st instance of two TempConvBlocks and ConvBlock
        k_tcb00 = ceil(self.rf[0]/self.dil[0])
        self.tcb00 = TempConvBlock(ch_in=4, ch_out=self.ch[0],
                                   k_size=k_tcb00, dil=self.dil[0],
                                   pad=((k_tcb00-1)*self.dil[0]+1)//2)
        k_tcb01 = ceil(self.rf[1]/self.dil[1])
        self.tcb01 = TempConvBlock(ch_in=self.ch[0], ch_out=self.ch[1],
                                   k_size=k_tcb01, dil=self.dil[1],
                                   pad=((k_tcb01-1)*self.dil[1]+1)//2)
        k_cb0 = ceil(self.rf[2]/self.dil[2])
        self.cb0 = ConvBlock(ch_in=self.ch[1], ch_out=self.ch[2],
                             k_size=k_cb0, s=1, dilation=self.dil[2],
                             pad=((k_cb0-1)*self.dil[2]+1)//2)

        # 2nd instance of two TempConvBlocks and ConvBlock
        k_tcb10 = ceil(self.rf[3]/self.dil[3])
        self.tcb10 = TempConvBlock(ch_in=self.ch[2], ch_out=self.ch[3],
                                   k_size=k_tcb10, dil=self.dil[3],
                                   pad=((k_tcb10-1)*self.dil[3]+1)//2)
        k_tcb11 = ceil(self.rf[4]/self.dil[4])
        self.tcb11 = TempConvBlock(ch_in=self.ch[3], ch_out=self.ch[4],
                                   k_size=k_tcb11, dil=self.dil[4],
                                   pad=((k_tcb11-1)*self.dil[4]+1)//2)
        self.cb1 = ConvBlock(ch_in=self.ch[4], ch_out=self.ch[5],
                             k_size=5, s=2, pad=2)

        # 3td instance of TempConvBlock and ConvBlock
        k_tcb20 = ceil(self.rf[5]/self.dil[5])
        self.tcb20 = TempConvBlock(ch_in=self.ch[5], ch_out=self.ch[6],
                                   k_size=k_tcb20, dil=self.dil[5],
                                   pad=((k_tcb20-1)*self.dil[5]+1)//2)
        k_tcb21 = ceil(self.rf[6]/self.dil[6])
        self.tcb21 = TempConvBlock(ch_in=self.ch[6], ch_out=self.ch[7],
                                   k_size=k_tcb21, dil=self.dil[6],
                                   pad=((k_tcb21-1)*self.dil[6]+1)//2)
        self.cb2 = ConvBlock(ch_in=self.ch[7], ch_out=self.ch[8],
                             k_size=5, s=4, pad=4)

        # 1st instance of regressor
        self.regr0 = Regressor(ft_in=self.ch[8] * 4, ft_out=self.ch[9])

        # 2nd instance of regressor
        self.regr1 = Regressor(ft_in=self.ch[9], ft_out=self.ch[10])

        # Output layer
        self.out_neuron = nn.Linear(in_features=self.ch[10], out_features=1)

    def forward(self, input):
        # 1st instance of two TempConvBlocks and ConvBlock
        x = self.tcb00(input)
        x = self.tcb01(x)
        x = self.cb0(x)
        # 2nd instance of two TempConvBlocks and ConvBlock
        x = self.tcb10(x)
        x = self.tcb11(x)
        x = self.cb1(x)
        # 3td instance of TempConvBlock and ConvBlock
        x = self.tcb20(x)
        x = self.tcb21(x)
        x = self.cb2(x)
        # Flatten
       # print(f" x1 = {x.shape}")
        x = x.flatten(1)
       # print(f" x2 = {x.shape}")
        # 1st instance of regressor
        x = self.regr0(x)
        # 2nd instance of regressor
        x = self.regr1(x)
        # Output layer
        x = self.out_neuron(x)
        return x

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
  plt.savefig(f'./Benchmark_hr_detection/pytorch_benchmarks/imgs/{type}/audio{num_sample}_epoch{epoch}.png') 
  
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
  plt.savefig(f'./Benchmark_hr_detection/pytorch_benchmarks/imgs/{type}/specto{num_sample}_epoch{epoch}.png') 

def unpatchify(imgs, type, dataset):
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
        if dataset == "DALIA" or dataset == "WESAD":
          x = imgs.reshape(shape=(imgs.shape[0], h, w, p, p, 4))
          x = torch.einsum('nhwpqc->nchpwq', x)
          specs = x.reshape(shape=(x.shape[0], 4, h * p, w * p))
        else: #ieeeppg
          x = imgs.reshape(shape=(imgs.shape[0], h, w, p, p, 5))
          x = torch.einsum('nhwpqc->nchpwq', x)
          specs = x.reshape(shape=(x.shape[0], 5, h * p, w * p))

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


class CheckPoint():
    """
    save/load a checkpoint based on a metric
    """
    def __init__(self, dir, net, optimizer, mode='min', fmt='ck_{epoch:03d}.pt'):
        if mode not in ['min', 'max']:
            raise ValueError("Early-stopping mode not supported") 
        self.dir = pathlib.Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.format = fmt
        self.net = net
        self.optimizer = optimizer
        self.val = None
        self.epoch = None
        self.best_path = None

    def __call__(self, epoch, val):
        val = float(val)
        if self.val == None:
            self.update_and_save(epoch, val)
        elif self.mode == 'min' and val < self.val:
            self.update_and_save(epoch, val)
        elif self.mode == 'max' and val > self.val:
            self.update_and_save(epoch, val)

    def update_and_save(self, epoch, val):
        self.epoch = epoch
        self.val = val
        self.update_best_path()
        self.save()

    def update_best_path(self):
        self.best_path = self.dir / self.format.format(**self.__dict__)

    def save(self, path=None):
        if path is None:
            path = self.best_path
        torch.save({
                  'epoch': self.epoch,
                  'model_state_dict': self.net.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict(),
                  'val': self.val,
                  }, path)

    def load_best(self):
        if self.best_path is None:
            raise FileNotFoundError("Best path not set!")
        self.load(self.best_path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def calculate_ae_accuracy(y_pred, y_true):
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.0, .01) * (np.amax(y_pred) - np.amin(y_pred))
    accuracy = 0
    for threshold in thresholds:
        y_pred_binary = (y_pred > threshold).astype(int)
        correct = np.sum(y_pred_binary == y_true)
        accuracy_tmp = 100 * correct / len(y_pred_binary)
        if accuracy_tmp > accuracy:
            accuracy = accuracy_tmp
    return accuracy


def calculate_ae_pr_accuracy(y_pred, y_true):
    # initialize all arrays
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.0, .01) * (np.amax(y_pred) - np.amin(y_pred))
    accuracy = 0
    n_normal = np.sum(y_true == 0)
    precision = np.zeros(len(thresholds))
    recall = np.zeros(len(thresholds))

    # Loop on all the threshold values
    for threshold_item in range(len(thresholds)):
        threshold = thresholds[threshold_item]
        # Binarize the result
        y_pred_binary = (y_pred > threshold).astype(int)
        # Build matrix of TP, TN, FP and FN
        false_positive = np.sum((y_pred_binary[0:n_normal] == 1))
        true_positive = np.sum((y_pred_binary[n_normal:] == 1))
        false_negative = np.sum((y_pred_binary[n_normal:] == 0))
        # Calculate and store precision and recall
        precision[threshold_item] = true_positive / (true_positive + false_positive)
        recall[threshold_item] = true_positive / (true_positive + false_negative)
        # See if the accuracy has improved
        accuracy_tmp = 100 * (precision[threshold_item] + recall[threshold_item]) / 2
        if accuracy_tmp > accuracy:
            accuracy = accuracy_tmp
    return accuracy


def calculate_ae_auc(y_pred, y_true):
    """
    Autoencoder ROC AUC calculation
    """
    # initialize all arrays
    thresholds = np.amin(y_pred) + np.arange(0.0, 1.01, .01) * (np.amax(y_pred) - np.amin(y_pred))
    roc_auc = 0

    n_normal = np.sum(y_true == 0)
    tpr = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))

    # Loop on all the threshold values
    for threshold_item in range(1, len(thresholds)):
        threshold = thresholds[threshold_item]
        # Binarize the result
        y_pred_binary = (y_pred > threshold).astype(int)
        # Build TP and FP
        tpr[threshold_item] = np.sum((y_pred_binary[n_normal:] == 1)
                                     ) / float(len(y_true) - n_normal)
        fpr[threshold_item] = np.sum((y_pred_binary[0:n_normal] == 1)) / float(n_normal)

    # Force boundary condition
    fpr[0] = 1
    tpr[0] = 1

    # Integrate
    for threshold_item in range(len(thresholds) - 1):
        roc_auc += .5 * (tpr[threshold_item] + tpr[threshold_item + 1]) * (
            fpr[threshold_item] - fpr[threshold_item + 1])
    return roc_auc


def seed_all(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    return seed

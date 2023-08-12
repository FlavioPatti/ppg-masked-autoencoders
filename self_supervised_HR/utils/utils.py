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

def get_reference_model(model_name: str):
    if model_name == 'temponet':
        return TEMPONet()

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

class Data_Augmentation:
    def __init__(self, X, y, augmentations = {}, datadir = 'None', groups = [], n_folds = 4, test_print = False):
        self.X = X
        self.y = y
        self.augmentations = augmentations
        self.datadir = datadir
        self.groups = groups
        self.n_folds = n_folds
        self.test_print = test_print

    def run(self):
        X_original = copy.deepcopy(self.X)
        y_original = copy.deepcopy(self.y)
        X_out = copy.deepcopy(self.X)
        y_out = copy.deepcopy(self.y)
        seed = 42
        #Application data augmentation
        import matplotlib.pyplot as plt 
        fig, axs = plt.subplots(4, 2)
        time_vector = np.linspace(0,8,256)
        i = 0
        i1 = 0
        j = 0
        for augment_key in self.augmentations:
            unique_augment = augment_key.split('+')
            np.random.seed(seed)
            shuffler = np.random.permutation(len(X_original))
            X_aug = X_original[shuffler]
            y_aug = y_original[shuffler]
            X_aug = X_aug[:int(self.augmentations[augment_key]['percentage']*len(X_aug))]
            y_aug = y_aug[:int(self.augmentations[augment_key]['percentage']*len(y_aug))]
            colors = ['C0', 'C1','C2','C3','C4','C5','C6']
            for single_augmentation in unique_augment:
                for key in Augmentations_list:
                    if key in single_augmentation:
                        X_aug1, y_aug1 = Augmentations_list[key](self, X_aug, y_aug, self.augmentations[augment_key])
                        if "Frequency_mul_up_to_2" in single_augmentation and self.test_print == True:
                            try:
                                axs[i1,j].plot(time_vector, X_aug1[0,0,:], color = colors[i], linewidth = '1',  label = f'{key}')
                            except:
                                pdb.set_trace()
                            axs[i1,j].grid()
                            axs[i1,j].set_ylim([-250,200])
                            i+=1
                            i1=i%4
                            j=i//4
                        if X_aug1.shape[0] > X_aug.shape[0]:
                            np.random.seed(seed)
                            shuffler = np.random.permutation(len(X_aug1))
                            X_aug1 = X_aug1[shuffler]
                            y_aug1 = y_aug1[shuffler]
                            X_aug1 = X_aug1[:int(self.augmentations[augment_key]['percentage']*len(X_aug1))]
                            y_aug1 = y_aug1[:int(self.augmentations[augment_key]['percentage']*len(y_aug1))] 
                        if "Frequency_mul_up_to_2" not in single_augmentation and self.test_print == True:

                            axs[0,0].plot(time_vector,X_aug[0,0,:], color = 'C0', linewidth = '1',  label = 'No Augment')
                            axs[0,0].grid()
                            axs[0,0].set_ylim([-250,200])
                            if i == 0:
                                i+=1
                                i1=1
                            axs[i1,j].plot(time_vector,X_aug1[0,0,:], color = colors[i], linewidth = '1',  label = f'{key}')
                            axs[i1,j].grid()
                            axs[i1,j].set_ylim([-250,200])
                            i+=1
                            i1=i%4
                            j=i//4
            X_out = np.concatenate((X_out, X_aug1))
            y_out = np.concatenate((y_out, y_aug1))
        plt.savefig('prova.png', dpi = 800)
        return X_out, y_out

    def Jittering(self, X, y, args):
        X1 = np.full_like(X, 0).astype('float32')
        # print(f"{X1.dtype} {X.dtype}")
        for i in range(X.shape[0]):
            for k in range(X.shape[1]):
                myNoise = np.random.normal(loc=0, scale=args['sigma']*max(abs(X[i][k])), size=(X.shape[2]))
                X1[i][k] = X[i][k] + myNoise
        return X1, y

    def Scaling(self, X, y, args):
        scalingFactor = np.random.normal(loc=1.0, scale=args['sigma'], size=(X.shape[2]))
        myNoise = (np.ones((X.shape[2])) * scalingFactor).astype('float32')
        return X * myNoise, y

    #Magnitude Warping (where X is training set)
    def DA_MagWarp(self, X, y, args):
        X1 = np.full_like(X, 0).astype('float32')
        from scipy.interpolate import CubicSpline
        xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[2], (X.shape[2]-1)/(args['knot']+1)))).transpose()
        yy = np.random.normal(loc=1.0, scale=args['sigma'], size=(args['knot']+2, X.shape[1]))
        x_range = np.arange(X.shape[2])
        cs_ppg = CubicSpline(xx[:,0], yy[:,0])
        cs_x = CubicSpline(xx[:,1], yy[:,1])
        cs_y = CubicSpline(xx[:,2], yy[:,2])
        cs_z = CubicSpline(xx[:,3], yy[:,3])
        generare_curve = np.array([cs_ppg(x_range),cs_x(x_range),cs_y(x_range),cs_z(x_range)])
        for i in range(X.shape[0]):
            X1[i] = X[i] * generare_curve
        return X1, y

    #Time Warping
    def DA_TimeWarp(self, X, y, args):
         from scipy.interpolate import CubicSpline
         xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[2], (X.shape[2]-1)/(args['knot']+1)))).transpose()
         yy = np.random.normal(loc=1.0, scale=args['sigma'], size=(args['knot']+2, 1))*np.ones((args['knot']+2, X.shape[1]))
         x_range = np.arange(X.shape[2])
         cs_ppg = CubicSpline(xx[:,0], yy[:,0])
         cs_x = CubicSpline(xx[:,1], yy[:,1])
         cs_y = CubicSpline(xx[:,2], yy[:,2])
         cs_z = CubicSpline(xx[:,3], yy[:,3])
         tt = np.array([cs_ppg(x_range),cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose() # Regard these samples aroun 1 as time intervals
         tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
          # Make the last value
         t_scale = [(X.shape[2]-1)/tt_cum[-1,0],(X.shape[2]-1)/tt_cum[-1,1],(X.shape[2]-1)/tt_cum[-1,2],(X.shape[2]-1)/tt_cum[-1,3]]
         tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
         tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
         tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
         tt_cum[:,3] = tt_cum[:,3]*t_scale[3]
         X_new = np.zeros(X.shape).astype('float32')
         x_range = np.arange(X.shape[2])
         for i in range(X.shape[0]):
             X_new[i][0] = np.interp(x_range, tt_cum[:,0], X[i][0])
             X_new[i][1] = np.interp(x_range, tt_cum[:,1], X[i][1])
             X_new[i][2] = np.interp(x_range, tt_cum[:,2], X[i][2])
             X_new[i][3] = np.interp(x_range, tt_cum[:,3], X[i][3])
         return X_new, y

    #frequencies divided by 2(where X is the training set)
    def Frequency_div_2(self, X, y, args):
         x = np.arange(X.shape[2])
         xvals = np.linspace(0, 255, num=512)
         x_interpolated = np.zeros((X.shape[0],X.shape[1],2*X.shape[2]))
         X_new = np.zeros(X.shape).astype('float32')
         #Oversampling the signal
         for i in range(X.shape[0]):
             for k in range(X.shape[1]):
                 x_interpolated[i][k] = np.interp(xvals, x, X[i][k])
        #Make two 4 second window
         for i in range(X.shape[0]):
             for k in range(X.shape[1]):
                 X_new[i][k] = x_interpolated[i][k][0:256]
         y_new = y/2
         return X_new, y_new

    #frequencies multiplied by two (where X is 16-second windows of the training set )
    ################################################################
    #### RESULTS COULD BE DIFFERENT SINCE I CHANGED FROM SELF.X TO X
    ################################################################
    def Frequency_mul_up_to_2(self, X, y, args):
         '''
         #By data augmentation frequencies multiplied by two
          x_16s = data_augmentation.x(self.data_dir)
          new_y = data_augmentation.y(x_16s,self._X,self._y)
          groups = data_augmentation.groups(x_16s,self._X,self._groups).astype('int64')
          x_8s= data_augmentation.new_x(x_16s,self._X)
          indices1, _ = self._rndgroup_kfold(groups, n
          y_16s = data_augmentation.label_16(self._y,x_16s,self._X)
          train_index, _ = indices1[fold]
         '''
         x_16s, y_16s = self.x_16s_creation(self.datadir, self.X, self.y, args)
         if args["multiplier"] > 1.9:
            X_out = x_16s[:,:,::2]
         else:
             X_out = np.zeros((x_16s.shape[0],x_16s.shape[1],256)).astype('float32')
            #create a 16 second window with 256 sample
             for i in range(x_16s.shape[0]):
                for k in range(x_16s.shape[1]):
                    from scipy.interpolate import interp1d
                    values = x_16s[i][k][:int(x_16s.shape[2]*args["multiplier"]/2)]
                    x_indexes = np.linspace(0, 256, num=int(x_16s.shape[2]*args["multiplier"]/2), endpoint=True)
                    f = interp1d(x_indexes, values, kind='cubic')
                    x_indexes_new = np.linspace(0, 256, num=256, endpoint=True)
                    X_out[i][k] = f(x_indexes_new).astype('float32')
         return X_out, y_16s
     
    def x_16s_creation(self, datadir, X, y, args):
         with open(datadir / 'slimmed_dalia_16.pkl', 'rb') as f:
             dataset = pickle.load(f, encoding='latin1')
         X_16, _, _ = dataset.values()
         indexes_train_16s = []
         y_16s = []
         i = 0
         for index_8s in np.arange(self.X.shape[0]):
            for index_16s in np.arange(i, X_16.shape[0]):
                if np.array_equal(self.X[index_8s][0][:],X_16[index_16s][0][:256]):
                    indexes_train_16s.append(index_16s)
                    y_16s.append(self.y[index_8s]*args["multiplier"])
                    i = index_16s
                    break
         return X_16[indexes_train_16s], np.asarray(y_16s)

Augmentations_list = {
    "Jittering": Data_Augmentation.Jittering, 
    "Scaling": Data_Augmentation.Scaling, 
    "DA_MagWarp": Data_Augmentation.DA_MagWarp, 
    "DA_TimeWarp": Data_Augmentation.DA_TimeWarp, 
    "Frequency_div_2": Data_Augmentation.Frequency_div_2, 
    "Frequency_mul_up_to_2": Data_Augmentation.Frequency_mul_up_to_2}
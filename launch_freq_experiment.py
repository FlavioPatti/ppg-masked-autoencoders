import torch
import wandb
import self_supervised_HR.freq as hrd
import self_supervised_HR.utils.utils as utils
from self_supervised_HR.utils import  EarlyStopping
from self_supervised_HR.freq.data import Dalia
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import sys
import os
from pathlib import Path
import pickle
from thop import profile
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

os.environ["WANDB_API_KEY"] = "20fed903c269ff76b57d17a58cdb0ba9d0c4d2be"
os.environ["WANDB_MODE"] = "online"

N_PRETRAIN_EPOCHS = 1
N_FINETUNE_EPOCHS = 1
K_FOLD = False
DATASET = "DALIA"

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

def save_checkpoint_pretrain(state, filename="checkpoint_model_pretrain"):
  print("=> Saving pretrained checkpoint")
  torch.save(state,filename)

def load_checkpoint_pretrain(checkpoint):
  print("=> Loading pretrained checkpoint")
  model.load_state_dict(checkpoint['state_dict'])
  
# Set earlystop
earlystop = EarlyStopping(patience=20, mode='min')
# Training Loop
loss_scaler = NativeScaler()

# Get the Model
print("=> Running Freq experiment")
model = utils.get_reference_model('vit_freq_pretrain') #ViT (encoder + decoder)

if torch.cuda.is_available():
  model = model.cuda()

# Get Training Settings
criterion = utils.get_default_criterion("pretrain")
optimizer = utils.get_default_optimizer(model, "pretrain")

if K_FOLD: #for time/freq experiments

  # Get the Data and perform cross-validation
  data_gen = hrd.get_data(dataset = DATASET)
  for datasets in data_gen:

    print(f"Using K-Fold")
    train_ds, val_ds, test_ds = datasets
    test_subj = test_ds.test_subj
    dataloaders = hrd.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders
    best_val_loss = sys.float_info.max

    #Pretraining for recostruct input signals
    for epoch in range(N_PRETRAIN_EPOCHS):
      
      train_stats = hrd.train_one_epoch_masked_autoencoder_freq(
          model, train_dl, criterion,
          optimizer, device, epoch, loss_scaler,
          normalization = True,
          plot_heatmap = False, 
          sample_to_plot = 50)

      val_stats = hrd.train_one_epoch_masked_autoencoder_freq(
          model, val_dl, criterion,
          optimizer, device, epoch, loss_scaler,
          normalization = True,
          plot_heatmap = False, 
          sample_to_plot = 50)

      test_stats = hrd.train_one_epoch_masked_autoencoder_freq(
          model, test_dl, criterion,
          optimizer, device, epoch, loss_scaler,
          normalization = True,
          plot_heatmap = False, 
          sample_to_plot = 50)

      print(f"train_stats = {train_stats}")
      print(f"val_stats = {val_stats}")
      print(f"test_stats = {test_stats}")
      
      val_loss = test_stats['loss']
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f"=> new best val loss found = {best_val_loss}")
        #Save checkpoint
        checkpoint = {'state_dict': model.state_dict()}
        save_checkpoint_pretrain(checkpoint)
      
      if earlystop(val_loss):
        break

    #Finetune for hr estimation
    model = utils.get_reference_model('vit_freq_finetune') #ViT (only encoder with at the end linear layer)
      
    #print #params and #ops for the model
    #input_tensor = torch.randn(1,4,64,256)
    #flops, params = profile(model, inputs=(input_tensor,))
    #print(f"# params = {params}, #flops = {flops}")

    if torch.cuda.is_available():
        model = model.cuda()
          
    # Get Training Settings
    criterion = utils.get_default_criterion("finetune")
    #optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    #print(f"optimizer => {optimizer}")
    #scheduler = StepLR(optimizer, step_size=20, gamma=1/3)
    optimizer = utils.get_default_optimizer(model, "finetune")
    earlystop = EarlyStopping(patience=20, mode='min')
    #Load checkpoint from pretrain if exists
    load_checkpoint_pretrain(torch.load("./checkpoint_model_pretrain"))
    best_val_mae = sys.float_info.max
    best_test_mae = sys.float_info.max

    for epoch in range(N_FINETUNE_EPOCHS):
      metrics = hrd.train_one_epoch_hr_detection_freq(
            epoch, model, criterion, optimizer, train_dl, val_dl, device,
            normalization = False,plot_heatmap = False, sample_to_plot = 50)
        
      print(f"train and val stats = {metrics}")
      val_mae = metrics['val_MAE']
      
      test_metrics = hrd.evaluate_freq(model, criterion, test_dl, device,
          normalization = False,plot_heatmap = False, sample_to_plot = 50)
        
      print(f"test stats = {test_metrics}")
      test_mae = test_metrics['MAE']

      if val_mae < best_val_mae:
        best_val_mae = val_mae
        print(f"new best val mae found = {best_val_mae}")
      
      if test_mae < best_test_mae:
        best_test_mae = test_mae
        print(f"new best test mae found = {best_test_mae}")

      #if epoch >= 30: #delayed earlystop
      if earlystop(val_mae):
        break

else: #for transfer learning
  print(f"Using Single Fold")
  data_dir = Path('.').absolute() / DATASET
  with open(data_dir / 'slimmed_dalia.pkl', 'rb') as f:
      ds = pickle.load(f, encoding='latin1')
  samples, target, groups = ds.values()
  data = Dalia(samples, target)
  train_dl = DataLoader(
      data,
      batch_size=128,
      shuffle=True,
      pin_memory=True,
      num_workers=4)

  best_loss = sys.float_info.max
  
  #Pretraining for recostruct input signals
  for epoch in range(N_PRETRAIN_EPOCHS):
    
    train_stats = hrd.train_one_epoch_masked_autoencoder_freq(
          model, train_dl, criterion,
          optimizer, device, epoch, loss_scaler,
          normalization = True,
          plot_heatmap = False, 
          sample_to_plot = 50)
    
    print(f"train_stats = {train_stats}")
    
    loss = train_stats['loss']
    if loss < best_loss:
      best_loss = loss
      print(f"=> new best loss found = {best_loss}")
      #Save checkpoint
      checkpoint = {'state_dict': model.state_dict()}
      save_checkpoint_pretrain(checkpoint)
    
    if earlystop(loss):
      break

  #Finetune for hr estimation
  model = utils.get_reference_model('vit_freq_finetune') #ViT (only encoder with at the end linear layer)
    
  #print #params and #ops for the model
  #input_tensor = torch.randn(1,4,64,256)
  #flops, params = profile(model, inputs=(input_tensor,))
  #print(f"# params = {params}, #flops = {flops}")

  if torch.cuda.is_available():
      model = model.cuda()
        
  # Get Training Settings
  criterion = utils.get_default_criterion("finetune")
  #optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
  #print(f"optimizer => {optimizer}")
  #scheduler = StepLR(optimizer, step_size=20, gamma=1/3)
  optimizer = utils.get_default_optimizer(model, "finetune")
  earlystop = EarlyStopping(patience=20, mode='min')
  #Load checkpoint from pretrain if exists
  load_checkpoint_pretrain(torch.load("./checkpoint_model_pretrain"))

  # Get the Data and perform cross-validation
  data_gen = hrd.get_data(dataset = DATASET)
  for datasets in data_gen:

    train_ds, val_ds, test_ds = datasets
    test_subj = test_ds.test_subj
    dataloaders = hrd.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders

    best_val_mae = sys.float_info.max
    best_test_mae = sys.float_info.max

    for epoch in range(N_FINETUNE_EPOCHS):
      metrics = hrd.train_one_epoch_hr_detection_freq(
          epoch, model, criterion, optimizer, train_dl, val_dl, device,
          normalization = False,plot_heatmap = False, sample_to_plot = 50)
      
    print(f"train and val stats = {metrics}")
    val_mae = metrics['val_MAE']

    test_metrics = hrd.evaluate_freq(model, criterion, test_dl, device,
        normalization = False,plot_heatmap = False, sample_to_plot = 50)
      
    print(f"test stats = {test_metrics}")
    test_mae = test_metrics['MAE']

    if val_mae < best_val_mae:
      best_val_mae = val_mae
      print(f"new best val mae found = {best_val_mae}")

    if test_mae < best_test_mae:
      best_test_mae = test_mae
      print(f"new best test mae found = {best_test_mae}")

    #if epoch >= 30: #delayed earlystop
    if earlystop(val_mae):
      break


      
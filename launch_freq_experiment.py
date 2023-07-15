import torch
import wandb
import numpy as np
import self_supervised_HR.freq.data_loader_ieee as load_ieee
import self_supervised_HR.freq as hrd
import self_supervised_HR.utils.utils as utils
from self_supervised_HR.utils import  EarlyStopping
from self_supervised_HR.freq.data_loader_dalia_wesad import Dalia
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import sys
import os
from pathlib import Path
import pickle
from thop import profile
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import csv
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split

os.environ["WANDB_API_KEY"] = "20fed903c269ff76b57d17a58cdb0ba9d0c4d2be"
os.environ["WANDB_MODE"] = "online"

N_PRETRAIN_EPOCHS = 1
N_FINETUNE_EPOCHS = 1
TRANSFER_LEARNING = False
DATASET_PRETRAIN = "DALIA"
DATASET_FINETUNING = "DALIA"

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
model = utils.get_reference_model('vit_freq_pretrain') #ViT (encoder + decoder)

if torch.cuda.is_available():
  model = model.cuda()

# Get Training Settings
criterion = utils.get_default_criterion("pretrain")
optimizer = utils.get_default_optimizer(model, "pretrain")

if not TRANSFER_LEARNING: #for time/freq experiments
  print(f"=> Running frequency experiment with dataset = {DATASET_PRETRAIN}")
  # Get the Data and perform cross-validation
  data_gen = hrd.get_data(dataset = DATASET_PRETRAIN)
  for datasets in data_gen:
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
      
      print(f"train stats = {train_stats}")
      print(f"val stats = {val_stats}")
      val_loss = val_stats['loss']
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

    for epoch in range(N_FINETUNE_EPOCHS):
      metrics = hrd.train_one_epoch_hr_detection_freq(
            epoch, model, criterion, optimizer, train_dl, val_dl, device,
            normalization = False,plot_heatmap = False, sample_to_plot = 50)
        
      print(f"train stats = {metrics}")
      val_mae = metrics['val_MAE']
      if val_mae < best_val_mae:
        best_val_mae = val_mae
        print(f"new best val mae found = {best_val_mae}")

      #if epoch >= 30: #delayed earlystop
      if earlystop(val_mae):
        break
      
    test_metrics = hrd.evaluate_freq(model, criterion, test_dl, device,
          normalization = False,plot_heatmap = False, sample_to_plot = 50)
    print(f"test stats = {test_metrics}")

else: #for transfer learning
  print(f"=> Running transfer learning experiment with pretrain dataset = {DATASET_PRETRAIN} and finetuning dataset = {DATASET_FINETUNING}")
  
  if DATASET_PRETRAIN == "DALIA" or DATASET_PRETRAIN == "WESAD":
    # Retrive the entire dataset
    data_dir = Path('.').absolute() / DATASET_PRETRAIN
    with open(data_dir / 'slimmed_dalia.pkl', 'rb') as f:
        ds = pickle.load(f, encoding='latin1')
    samples, target, groups = ds.values()
    full_dataset = Dalia(samples, target)
    train_dl = DataLoader(
        full_dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        num_workers=4)
  else: #IEEEPPG 
    # set data folder, train & test
    data_folder = "./IEEEPPG/"
    train_file = data_folder + "IEEEPPG_TRAIN.ts"
    test_file = data_folder + "IEEEPPG_TEST.ts"
    norm = "none"
    
    # loading the data. X_train and X_test are dataframe of N x n_dim
    X_train, y_train = load_ieee.load_from_tsfile_to_dataframe(train_file)
    X_test, y_test = load_ieee.load_from_tsfile_to_dataframe(test_file)

   # in case there are different lengths in the dataset, we need to consider that all the dimensions are the same length
    min_len = np.inf
    for i in range(len(X_train)):
        x = X_train.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)
    for i in range(len(X_test)):
        x = X_test.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)

    # process the data into numpy array
    x_train = load_ieee.process_data(X_train, normalise=norm, min_len=min_len)
    x_train = np.transpose(x_train, (0, 2, 1))
    x_test = load_ieee.process_data(X_test, normalise=norm, min_len=min_len)
    x_test = np.transpose(x_test, (0, 2, 1))
    print(f"x train = {x_train.shape}")
    
    #retrive training and validation from training data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    print(f"x train = {x_train.shape}")
    print(f"x val = {x_val.shape}")
    print(f"y train = {y_train.shape}")
    print(f"y val = {y_val.shape}")

    # Ridimensionamento del set di dati di convalida
    x_val = x_val[:100]
    y_val = y_val[:100]
        
    train_dl, val_dl, test_dl = load_ieee.get_dataloaders(x_train, x_val, x_test, y_train, y_val, y_test)
    
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
  data_gen = hrd.get_data(dataset = DATASET_FINETUNING)
  for datasets in data_gen:
    train_ds, val_ds, test_ds = datasets
    test_subj = test_ds.test_subj
    dataloaders = hrd.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders
    best_val_mae = sys.float_info.max

    for epoch in range(N_FINETUNE_EPOCHS):
      metrics = hrd.train_one_epoch_hr_detection_freq(
          epoch, model, criterion, optimizer, train_dl, val_dl, device,
          normalization = False,plot_heatmap = False, sample_to_plot = 50)
      
      print(f"train and val stats = {metrics}")
      val_mae = metrics['val_MAE']
      if val_mae < best_val_mae:
        best_val_mae = val_mae
        print(f"new best val mae found = {best_val_mae}")
      
      #if epoch >= 30: #delayed earlystop
      if earlystop(val_mae):
        break

    test_metrics = hrd.evaluate_freq(model, criterion, test_dl, device,
        normalization = False,plot_heatmap = False, sample_to_plot = 50)  
    print(f"test stats = {test_metrics}")
      
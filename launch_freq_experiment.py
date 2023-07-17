import torch
import wandb
import numpy as np
import self_supervised_HR.freq as hrd
import self_supervised_HR.utils.utils as utils
from self_supervised_HR.utils import  EarlyStopping
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import sys
import os
from thop import profile

#Wandb setup
os.environ["WANDB_API_KEY"] = "20fed903c269ff76b57d17a58cdb0ba9d0c4d2be"
os.environ["WANDB_MODE"] = "online"

# Set flags for experiments
N_PRETRAIN_EPOCHS = 1
N_FINETUNE_EPOCHS = 1
TRANSFER_LEARNING = True
DATASET_PRETRAIN = "IEEEPPG"
DATASET_FINETUNING = "DALIA"

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

if not TRANSFER_LEARNING: #for time/freq experiments
  print(f"=> Running frequency experiment with dataset = {DATASET_PRETRAIN}")
  
  # Get the Data and perform cross-validation
  data_gen = hrd.get_data(dataset = DATASET_PRETRAIN)
  for datasets in data_gen:
    train_ds, val_ds, test_ds = datasets
    test_subj = test_ds.test_subj
    dataloaders = hrd.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders
    
    # Set earlystop
    earlystop = EarlyStopping(patience=20, mode='min')
    # Set loss scaler for pretrain 
    loss_scaler = NativeScaler()

    # Get the Model
    model = utils.get_reference_model('vit_freq_pretrain', DATASET_PRETRAIN) #ViT (encoder + decoder)
    if torch.cuda.is_available():
      model = model.cuda()

    # Get Training Settings
    criterion = utils.get_default_criterion("pretrain")
    optimizer = utils.get_default_optimizer(model, "pretrain")
    best_loss = sys.float_info.max

    print(f"=> Starting pretrain for {N_PRETRAIN_EPOCHS} epochs...")
    #Pretraining for recostruct input signals
    for epoch in range(N_PRETRAIN_EPOCHS):
      
      train_stats = hrd.train_one_epoch_masked_autoencoder_freq(
          model, train_dl, criterion,
          optimizer, device, epoch, loss_scaler,
          normalization = True,
          plot_heatmap = False, 
          sample_to_plot = 50,
          dataset_name = DATASET_PRETRAIN)

      print(f"train stats = {train_stats}")
      loss = train_stats['loss']
      if loss < best_loss:
        best_loss = loss
        print(f"=> new best val loss found = {best_loss}")
        #Save checkpoint
        checkpoint = {'state_dict': model.state_dict()}
        utils.save_checkpoint_pretrain(checkpoint)
      
      if earlystop(loss):
        break
    print(f" => Done pretrain")
    
    # Set earlystop
    earlystop = EarlyStopping(patience=20, mode='min')
      
    #Finetune for hr estimation
    model = utils.get_reference_model('vit_freq_finetune') #ViT (only encoder with at the end linear layer)
    if torch.cuda.is_available():
        model = model.cuda()
      
    #print #params and #ops for the model
    #input_tensor = torch.randn(1,4,64,256)
    #flops, params = profile(model, inputs=(input_tensor,))
    #print(f"# params = {params}, #flops = {flops}")
          
    # Get Training Settings
    criterion = utils.get_default_criterion("finetune")
    #optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    #print(f"optimizer => {optimizer}")
    #scheduler = StepLR(optimizer, step_size=20, gamma=1/3)
    optimizer = utils.get_default_optimizer(model, "finetune")
    
    best_val_mae = sys.float_info.max
    best_test_mae = sys.float_info.max
    
    #Load checkpoint from pretrain if exists
    utils.load_checkpoint_pretrain(model, torch.load("./checkpoint_model_pretrain"))

    print(f"=> Starting finetuning for {N_FINETUNE_EPOCHS} epochs...")
    for epoch in range(N_FINETUNE_EPOCHS):
      train_metrics = hrd.train_one_epoch_hr_detection_freq(
            epoch, model, criterion, optimizer, train_dl, val_dl, device,
            normalization = False,plot_heatmap = False, sample_to_plot = 50)
      
      test_metrics = hrd.evaluate_freq(model, criterion, test_dl, device,
          normalization = False,plot_heatmap = False, sample_to_plot = 50)
        
      print(f"train stats = {train_metrics}")
      print(f"test stats = {test_metrics}")
      val_mae = train_metrics['val_MAE']
      if val_mae < best_val_mae:
        best_val_mae = val_mae
        print(f"new best val mae found = {best_val_mae}")
      test_mae = train_metrics['MAE']
      if test_mae < best_test_mae:
        best_test_mae = test_mae
        print(f"new best test mae found = {best_test_mae}")

      #if epoch >= 30: #delayed earlystop
      if earlystop(val_mae):
        break
      
    print(f" => Done finetuning")

else: #for transfer learning
  print(f"=> Running transfer learning experiment with pretrain dataset = {DATASET_PRETRAIN} and finetuning dataset = {DATASET_FINETUNING}")
  
  # Retrive the entire dataset
  train_dl = hrd.get_full_dataset(DATASET_PRETRAIN)
  
  # Set earlystop
  earlystop = EarlyStopping(patience=20, mode='min')
  # Set loss scaler for pretrain 
  loss_scaler = NativeScaler()

  # Get the Model
  model = utils.get_reference_model('vit_freq_pretrain', DATASET_PRETRAIN) #ViT (encoder + decoder)
  if torch.cuda.is_available():
    model = model.cuda()

  # Get Training Settings
  criterion = utils.get_default_criterion("pretrain")
  optimizer = utils.get_default_optimizer(model, "pretrain")
  best_loss = sys.float_info.max
  
  print(f"=> Starting pretrain for {N_PRETRAIN_EPOCHS} epochs...")
  #Pretraining for recostruct input signals
  for epoch in range(N_PRETRAIN_EPOCHS):
        
    train_stats = hrd.train_one_epoch_masked_autoencoder_freq(
          model, train_dl, criterion,
          optimizer, device, epoch, loss_scaler,
          normalization = True,
          plot_heatmap = False, 
          sample_to_plot = 50, 
          dataset_name = DATASET_PRETRAIN)
    
    print(f"train_stats = {train_stats}")
    loss = train_stats['loss']
    if loss < best_loss:
      best_loss = loss
      print(f"=> new best loss found = {best_loss}")
      #Save checkpoint
      checkpoint = {'state_dict': model.state_dict()}
      utils.save_checkpoint_pretrain(checkpoint)
    
    if earlystop(loss):
      break
  print(f"=> Done pretrain")
      
  #Load checkpoint from pretrain if exists
  utils.load_checkpoint_pretrain(model, torch.load("./checkpoint_model_pretrain"))

  # Get the Data and perform cross-validation
  data_gen = hrd.get_data(dataset = DATASET_FINETUNING)
  for datasets in data_gen:
    train_ds, val_ds, test_ds = datasets
    test_subj = test_ds.test_subj
    dataloaders = hrd.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders
  
    # Set earlystop
    earlystop = EarlyStopping(patience=20, mode='min')

    #Finetune for hr estimation
    model = utils.get_reference_model('vit_freq_finetune', DATASET_PRETRAIN) #ViT (only encoder with at the end linear layer)
    if torch.cuda.is_available():
        model = model.cuda()
    
    #print #params and #ops for the model
    #input_tensor = torch.randn(1,4,64,256)
    #flops, params = profile(model, inputs=(input_tensor,))
    #print(f"# params = {params}, #flops = {flops}")
        
    # Get Training Settings
    criterion = utils.get_default_criterion("finetune")
    #optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    #scheduler = StepLR(optimizer, step_size=20, gamma=1/3)
    optimizer = utils.get_default_optimizer(model, "finetune")
    best_val_mae = sys.float_info.max
    best_test_mae = sys.float_info.max

    print(f"=> Starting finetuning for {N_FINETUNE_EPOCHS} epochs...")
    for epoch in range(N_FINETUNE_EPOCHS):
      train_metrics = hrd.train_one_epoch_hr_detection_freq(
          epoch, model, criterion, optimizer, train_dl, val_dl, device,
          normalization = False,plot_heatmap = False, sample_to_plot = 50)
      
      test_metrics = hrd.evaluate_freq(model, criterion, test_dl, device,
        normalization = False,plot_heatmap = False, sample_to_plot = 50)  
      
      print(f"train and val stats = {train_metrics}")
      print(f"test stats = {test_metrics}")
      val_mae = train_metrics['val_MAE']
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
      
    print(f"=> Done finetuning")
  
      
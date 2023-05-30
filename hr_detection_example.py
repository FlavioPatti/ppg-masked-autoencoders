import torch
import wandb
import pytorch_benchmarks.hr_detection as hrd
from pytorch_benchmarks.utils import  EarlyStopping
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import sys
import os
from thop import profile

os.environ["WANDB_API_KEY"] = "20fed903c269ff76b57d17a58cdb0ba9d0c4d2be"
os.environ["WANDB_MODE"] = "online"

N_PRETRAIN_EPOCHS = 0
N_FINETUNE_EPOCHS = 1000

#Type of experiments: 
FREQ = 0
TIME = 1

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

def save_checkpoint_pretrain(state, filename="checkpoint_model_pretrain"):
  print("=> Saving checkpoint")
  torch.save(state,filename)

def load_checkpoint_pretrain(checkpoint):
  print("=> Loading checkpoint")
  model.load_state_dict(checkpoint['state_dict'])

def save_checkpoint_finetune(state, filename="checkpoint_model_finetune"):
  print("=> Saving checkpoint")
  torch.save(state,filename)

def load_checkpoint_finetune(checkpoint):
  print("=> Loading checkpoint")
  model.load_state_dict(checkpoint['state_dict'])
  
# Init wandb for plot loss/mae
configuration = {'experiment': "Time", 'epochs_pretrain': N_PRETRAIN_EPOCHS, 'epochs_finetune': N_FINETUNE_EPOCHS}
run = wandb.init(
            # Set entity to specify your username or team name
            entity = "aml-2022", 
            # Set the project where this run will be logged
            project="Hr_detection",
            group='finetuning1',
            # Track hyperparameters and run metadata
            config=configuration,
            resume="allow")
  
# Get the Data and perform cross-validation
mae_dict = dict()
data_gen = hrd.get_data()
for datasets in data_gen:
    train_ds, val_ds, test_ds = datasets
    test_subj = test_ds.test_subj
    dataloaders = hrd.build_dataloaders(datasets)
    train_dl, val_dl, test_dl = dataloaders
    # Set earlystop
    earlystop = EarlyStopping(patience=20, mode='min')
    # Training Loop
    loss_scaler = NativeScaler()
    
    # Get the Model
    if FREQ:
      print("Freq experiment")
      model = hrd.get_reference_model('vit_freq+time_pretrain') #ViT (encoder + decoder)
    if TIME:
      print("Time experiment")
      model = hrd.get_reference_model('vit_time_pretrain') #ViT (encoder + decoder)
      
    if torch.cuda.is_available():
      model = model.cuda()
    
    # Get Training Settings
    criterion = hrd.get_default_criterion("pretrain")
    optimizer = hrd.get_default_optimizer(model, "pretrain")
    best_loss = sys.float_info.max
    
    
    #Pretraining for recostruct input signals
    for epoch in range(N_PRETRAIN_EPOCHS):
        
      if FREQ:
        train_stats = hrd.train_one_epoch_masked_autoencoder_freq_time(
            model, train_dl, criterion,
            optimizer, device, epoch, loss_scaler,
            log_writer=None,
            args=None
        )

      if TIME:
        train_stats = hrd.train_one_epoch_masked_autoencoder_time(
            model, train_dl, criterion,
            optimizer, device, epoch, loss_scaler,
            log_writer=None,
            args=None
        )
      
      print(f"train_stats = {train_stats}")
      loss = train_stats['loss']
      if loss < best_loss:
        best_loss = loss
        print(f"new best loss found = {best_loss}")
        #Save checkpoint
        checkpoint = {'state_dict': model.state_dict()}
        save_checkpoint_pretrain(checkpoint)
      
      if earlystop(train_stats['loss']):
        break
    
    #Finetune for hr estimation
    if FREQ:
      model = hrd.get_reference_model('vit_freq+time_finetune') #ViT (only encoder with at the end linear layer)
    if TIME:
      model = hrd.get_reference_model('vit_time_finetune') #ViT (only encoder with at the end linear layer)

    input_tensor = torch.randn(1,4,256,1)
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"# params = {params}, #flops = {flops}")
      
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Get Training Settings
    criterion = hrd.get_default_criterion("finetune")
    optimizer = hrd.get_default_optimizer(model, "finetune")
    earlystop = EarlyStopping(patience=20, mode='min')
    best_loss = sys.float_info.max

    #Load checkpoint from pretrain if exists
    #load_checkpoint_pretrain(torch.load("./checkpoint_model_pretrain"))
    
    for epoch in range(N_FINETUNE_EPOCHS):
      if FREQ:
        metrics = hrd.train_one_epoch_hr_detection_freq_time(
            epoch, model, criterion, optimizer, train_dl, val_dl, device)
      if TIME:
        metrics = hrd.train_one_epoch_hr_detection_time(
            epoch, model, criterion, optimizer, train_dl, val_dl, device)
        
      print(f"train stats = {metrics}")
      train_loss = metrics['loss']
      train_mae = metrics['MAE']
      val_loss = metrics['val_loss']
      val_mae = metrics['val_MAE']
      
      if train_loss < best_loss:
        best_loss = train_loss
        print(f"new best loss found = {best_loss}")
        
      print(f"=> Updating plot on wandb")
      wandb.log({'train_loss': train_loss, 'epochs': epoch + 1}, commit=True)
      wandb.log({'train_mae': train_mae, 'epochs': epoch + 1}, commit=True)
      wandb.log({'val_loss': val_loss, 'epochs': epoch + 1}, commit=True)
      wandb.log({'val_mae': val_mae, 'epochs': epoch + 1}, commit=True)

      if earlystop(val_loss):
        break

      if FREQ:
        test_metrics = hrd.evaluate_freq_time(model, criterion, test_dl, device)
      if TIME:
        test_metrics = hrd.evaluate_time(model, criterion, test_dl, device)
        
      print(f"test stats = {test_metrics}")
      test_loss = test_metrics['loss']
      test_mae = test_metrics['MAE']
      
      print(f"=> Updating plot on wandb")
      wandb.log({'test_loss': test_loss, 'epochs': epoch + 1}, commit=True)
      wandb.log({'test_mae': test_mae, 'epochs': epoch + 1}, commit=True)
    
      
    
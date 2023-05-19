import torch
import wandb
import pytorch_benchmarks.hr_detection as hrd
from pytorch_benchmarks.utils import  EarlyStopping
from util.misc import NativeScalerWithGradNormCount as NativeScaler
N_PRETRAIN_EPOCHS = 0
N_FINETUNE_EPOCHS = 200

#Type of experiments: 
FREQ_PLUS_TIME = 1
TIME = 0

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
    if FREQ_PLUS_TIME:
      print("Freq+Time experiment")
      model = hrd.get_reference_model('vit_freq+time_pretrain') #ViT (encoder + decoder)
    if TIME:
      print("Time experiment")
      model = hrd.get_reference_model('vit_time_pretrain') #ViT (encoder + decoder)
    if torch.cuda.is_available():
      model = model.cuda()
    
    # Get Training Settings
    criterion = hrd.get_default_criterion("pretrain")
    optimizer = hrd.get_default_optimizer(model, "pretrain")
    best_loss = 1000
    
    # Init wandb for plot results
    configuration = {'experiment': "Time", 'epochs_pretrain': N_PRETRAIN_EPOCHS, 'epochs_finetune': N_FINETUNE_EPOCHS}
    run = wandb.init(
                # Set entity to specify your username or team name
                entity = "aml-2022", 
                # Set the project where this run will be logged
                project="Hr_detection",
                group='finetuning',
                # Track hyperparameters and run metadata
                config=configuration,
                resume="allow")
    
    #Pretraining for recostruct input signals
    for epoch in range(N_PRETRAIN_EPOCHS):
        
      if FREQ_PLUS_TIME:
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
    if FREQ_PLUS_TIME:
      model = hrd.get_reference_model('vit_freq+time_finetune') #ViT (only encoder with at the end linear layer)
    if TIME:
      model = hrd.get_reference_model('vit_time_finetune') #ViT (only encoder with at the end linear layer)
    if torch.cuda.is_available():
        model = model.cuda()
    # Get Training Settings
    criterion = hrd.get_default_criterion("finetune")
    optimizer = hrd.get_default_optimizer(model, "finetune")
    best_loss = 1000

    #Load checkpoint from pretrain if exists
    #load_checkpoint_pretrain(torch.load("./checkpoint_model_pretrain"))
    
    for epoch in range(N_FINETUNE_EPOCHS):
      if FREQ_PLUS_TIME:
        metrics = hrd.train_one_epoch_hr_detection_freq_time(
            epoch, model, criterion, optimizer, train_dl, val_dl, device)
      if TIME:
        metrics = hrd.train_one_epoch_hr_detection_time(
            epoch, model, criterion, optimizer, train_dl, val_dl, device)
        
      print(f"metrics = {metrics}")
      loss = metrics['loss']
      mae = metrics['MAE']
      print(f"=> Updating plot on wandb")
      wandb.log({'loss': loss, 'epochs': epoch + 1}, commit=True)
      wandb.log({'mae': mae, 'epochs': epoch + 1}, commit=True)
      
      if loss < best_loss:
        best_loss = loss
        print(f"new best loss found = {best_loss}")
      
      #if earlystop(metrics['val_MAE']):
       #   break
    
    #Evaluation on test dataset
    if FREQ_PLUS_TIME:
      test_metrics = hrd.evaluate_freq_time(model, criterion, test_dl, device)
    if TIME:
      test_metrics = hrd.evaluate_time(model, criterion, test_dl, device)
    print("Test Set Loss:", test_metrics['loss'])
    print("Test Set MAE:", test_metrics['MAE'])
    print(f"=> Updating plot on wandb")
    wandb.log({'test_loss': test_metrics['loss'], 'epochs': epoch + 1}, commit=True)
    wandb.log({'test_mae': test_metrics['MAE'], 'epochs': epoch + 1}, commit=True)
    mae_dict[test_subj] = test_metrics['MAE']
    print(f'MAE: {mae_dict}')
    print(f'Average MAE: {sum(mae_dict.values()) / len(mae_dict)}')

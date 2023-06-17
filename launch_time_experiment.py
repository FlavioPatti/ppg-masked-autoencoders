import torch
import wandb
import self_supervised_HR.time as hrd
import self_supervised_HR.utils.utils as utils
from self_supervised_HR.utils import  EarlyStopping
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import sys
import os
from thop import profile
from torch.optim.lr_scheduler import StepLR

os.environ["WANDB_API_KEY"] = "20fed903c269ff76b57d17a58cdb0ba9d0c4d2be"
os.environ["WANDB_MODE"] = "online"

N_PRETRAIN_EPOCHS = 0
N_FINETUNE_EPOCHS = 200

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

def save_checkpoint_pretrain(state, filename="checkpoint_model_pretrain"):
  print("=> Saving pretrained checkpoint")
  torch.save(state,filename)

def load_checkpoint_pretrain(checkpoint):
  print("=> Loading pretrained checkpoint")
  model.load_state_dict(checkpoint['state_dict'])
 
# Init wandb for plot loss/mae
configuration = {'experiment': "Time", 'epochs_pretrain': N_PRETRAIN_EPOCHS, 'epochs_finetune': N_FINETUNE_EPOCHS}
run = wandb.init(
            # Set entity to specify your username or team name
            entity = "aml-2022", 
            # Set the project where this run will be logged
            project="Hr_detection",
            group='finetuning2',
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
    print("=> Running Time experiment")
    model = utils.get_reference_model('vit_time_pretrain') #ViT (encoder + decoder)
      
    if torch.cuda.is_available():
      model = model.cuda()
    
    # Get Training Settings
    criterion = utils.get_default_criterion("pretrain")
    optimizer = utils.get_default_optimizer(model, "pretrain")
    best_loss = sys.float_info.max
    
    #Pretraining for recostruct input signals
    for epoch in range(N_PRETRAIN_EPOCHS):
    
      train_stats = hrd.train_one_epoch_masked_autoencoder_time(
          model, train_dl, criterion,
          optimizer, device, epoch, loss_scaler,
          normalization = False,
          plot_heatmap = False, 
          sample_to_plot = 50
      )
      
      print(f"train_stats = {train_stats}")
      loss = train_stats['loss']
      if loss < best_loss:
        best_loss = loss
        print(f"=> new best loss found = {best_loss}")
        #Save checkpoint
        checkpoint = {'state_dict': model.state_dict()}
        save_checkpoint_pretrain(checkpoint)
      
      if earlystop(train_stats['loss']):
        break
    
    #Finetune for hr estimation
    model = utils.get_reference_model('vit_time_finetune') #ViT (only encoder with at the end linear layer)

    #print #params and #ops for the model
    input_tensor = torch.randn(1,4,256,1) #(1,4,256,1) for time
    flops, params = profile(model, inputs=(input_tensor,))
    print(f"# params = {params}, #flops = {flops}")
  
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Get Training Settings
    criterion = utils.get_default_criterion("finetune")
    #optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    #print(f"optimizer => {optimizer}")
    #scheduler = StepLR(optimizer, step_size=20, gamma=1/3)
    optimizer = utils.get_default_optimizer(model, "finetune")
    earlystop = EarlyStopping(patience=20, mode='min')
    best_val_mae = sys.float_info.max
    best_test_mae = sys.float_info.max

    #Load checkpoint from pretrain if exists
    #load_checkpoint_pretrain(torch.load("./checkpoint_model_pretrain"))
    
    for epoch in range(N_FINETUNE_EPOCHS):
      metrics = hrd.train_one_epoch_hr_detection_time(
            epoch, model, criterion, optimizer, train_dl, val_dl, device,
            normalization = False,plot_heatmap = False, sample_to_plot = 50)
        
      print(f"train stats = {metrics}")
      train_mae = metrics['MAE']
      val_mae = metrics['val_MAE']
        
      print(f"=> Updating plot on wandb")
      wandb.log({'train_mae': train_mae, 'epochs': epoch + 1}, commit=True)
      wandb.log({'val_mae': val_mae, 'epochs': epoch + 1}, commit=True)

      test_metrics = hrd.evaluate_time(model, criterion, test_dl, device,
          normalization = False,plot_heatmap = False, sample_to_plot = 50)
        
      print(f"test stats = {test_metrics}")
      test_mae = test_metrics['MAE']

      if val_mae < best_val_mae:
        best_val_mae = val_mae
        print(f"new best val mae found = {best_val_mae}")
      
      print(f"=> Updating plot on wandb")
      wandb.log({'test_mae': test_mae, 'epochs': epoch + 1}, commit=True)
      
      if test_mae < best_test_mae:
        best_test_mae = test_mae
        print(f"new best test mae found = {best_test_mae}")

      if epoch >= 30: #delayed earlystop
        if earlystop(val_mae):
          break
      
import torch
#from pytorch_model_summary import summary
import pytorch_benchmarks.hr_detection as hrd
from pytorch_benchmarks.utils import seed_all, EarlyStopping
from util.misc import NativeScalerWithGradNormCount as NativeScaler
N_EPOCHS = 1

# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Ensure deterministic execution
#seed = seed_all(seed=42)

# Model Summary
#input_example = torch.rand((1,) + model.input_shape)
#print(summary(model, input_example.to(device), show_input=False, show_hierarchical=True))

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
    model = hrd.get_reference_model('vit_freq+time_pretrain') #vit or temponet
    if torch.cuda.is_available():
      model = model.cuda()
    # Get Training Settings
    criterion = hrd.get_default_criterion("pretrain")
    optimizer = hrd.get_default_optimizer(model, "pretrain")
    
    """pretraining for recostruct input signals"""
    for epoch in range(N_EPOCHS):
        
        train_stats = hrd.train_one_epoch_masked_autoencoder_freq_time(
            model, train_dl, criterion,
            optimizer, device, epoch, loss_scaler,
            log_writer=None,
            args=None
        )
        
        print(f" train_stats = {train_stats}")
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        
        print(f" log_stats = {log_stats}")
        
        if earlystop(log_stats['train_loss']):
            break
    
    #salvo i pesi del vecchio modello
    torch.save(model.state_dict(), "./pytorch_benchmarks/checkpoint")
    
    """finetune for hr estimation """
    # Get Training Settings
    criterion = hrd.get_default_criterion("finetune")
    optimizer = hrd.get_default_optimizer(model, "finetune")
    model = hrd.get_reference_model('vit_freq+time_finetune') #vit or temponet
    if torch.cuda.is_available():
        model = model.cuda()
    
    #loddo i pesi del vecchio modello al nuovo modello
    model.load_state_dict(torch.load("./pytorch_benchmarks/checkpoint"))
    
    for epoch in range(N_EPOCHS):
        metrics = hrd.train_one_epoch_hr_detection(
            epoch, model, criterion, optimizer, train_dl, val_dl, device)
        
        if earlystop(metrics['val_MAE']):
            break
        
    test_metrics = hrd.evaluate(model, criterion, test_dl, device)
    print("Test Set Loss:", test_metrics['loss'])
    print("Test Set MAE:", test_metrics['MAE'])
    mae_dict[test_subj] = test_metrics['MAE']

    print(f'MAE: {mae_dict}')
    print(f'Average MAE: {sum(mae_dict.values()) / len(mae_dict)}')
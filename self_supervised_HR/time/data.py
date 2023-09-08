from typing import Tuple
import pandas as pd
from pathlib import Path
import pickle
import random
import requests
import zipfile
import self_supervised_HR.freq as hrd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from skimage.util.shape import view_as_windows
import torch
from torch.utils.data import Dataset, DataLoader
import neurokit2
import wfdb.processing
import self_supervised_HR.utils as utils
import numpy as np
from statistics import mean

augmentations = {'Jittering': {'percentage': 0.9, 'sigma': 5/100},
                    'Scaling': {'percentage': 0.9, 'sigma': 0.3},
                    'DA_MagWarp': {'percentage': 0.9, 'sigma': 0.5, 'knot': 4},
                    'DA_TimeWarp': {'percentage': 0.9, 'sigma': 0.5, 'knot': 4},
                    'Frequency_div_2': {'percentage': 0.9} }

DALIA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00495/data.zip"
WESAD_URL = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"

def _collect_data(data_dir, data):
    random.seed(42)
    folder = ""
    if data == "WESAD":
      folder = "WESAD"
      #num = [2,3,4,5,6,7,8,9,10]
      num = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
    elif data == "DALIA":
      folder = "PPG_FieldStudy"
      num = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

    dataset = dict()
    session_list = random.sample(num, len(num))
    
    if data == "DALIA" or data == "WESAD":
      for subj in session_list:
        with open(data_dir / folder / f'S{str(subj)}' / f'S{str(subj)}.pkl', 'rb') as f:
            subject = pickle.load(f, encoding='latin1')
            ppg = subject['signal']['wrist']['BVP'][::2].astype('float32')
            acc = subject['signal']['wrist']['ACC'].astype('float32')
            ecg_signal = subject['signal']['chest']['ECG'].astype('float32')
            ecg_signal = np.squeeze(ecg_signal)
        if data == "DALIA":
            target = subject['label'].astype('float32')
        elif data == "WESAD":
            
            #Retrive heart rate from ecg_signal: first we extract peaks from ecg signal, we correct them and
            #then we use a sliding window of 8 seconds with shift of 2 seconds to compute the mean of istant heart rate
    
            fs = 700
            #Extract peaks
            _, results = neurokit2.ecg_peaks(ecg_signal, sampling_rate=fs)
            rpeaks = results["ECG_R_Peaks"]

            #Correct peaks
            rpeaks = wfdb.processing.correct_peaks(
            ecg_signal, rpeaks, search_radius=36, smooth_window_size=50, peak_dir="up")

            intervalli_tempo = [rpeaks[i+1] - rpeaks[i] for i in range(len(rpeaks)-1)]
            instant_heart_rate = [60 / (intervallo_tempo / fs) for intervallo_tempo in intervalli_tempo]
            window_size = 8.0 * fs 
            shift = 2.0 * fs  
        
            heart_rate_mean = []
            numero_iterazioni = int((ecg_signal.shape[0] - window_size) // shift + 1)
            for j in range(0, numero_iterazioni):
              heart_rate_current_window = []
              inizio_finestra = j * shift
              fine_finestra = inizio_finestra + window_size
              for i in range(0, len(rpeaks)-1):
                if rpeaks[i] >= inizio_finestra and rpeaks[i] < fine_finestra:
                  heart_rate_current_window.append(instant_heart_rate[i])
              heart_rate_mean.append(mean(heart_rate_current_window))
          
            target = np.array(heart_rate_mean).astype('float32')
        dataset[subj] = { 
        #each sample is build by: ppg value, accelerometer value, hr estimation
          'ppg': ppg,
          'acc': acc,
          'target': target
              }
        
    return dataset



def _preprocess_data(data_dir, dataset, dataset_name):
    """
    Process data with a sliding window of size 'time_window' and overlap 'overlap'
    """
    fs = 32
      
    time_window = 8
    overlap = 2

    groups = list()
    signals = list()
    targets = list()

    for k in dataset:
        sig = np.concatenate((dataset[k]['ppg'], dataset[k]['acc']), axis=1)
        sig = np.moveaxis(
            view_as_windows(sig, (fs*time_window, 4), fs*overlap)[:, 0, :, :],
            1, 2)
        groups.append(np.full(sig.shape[0], k))
        signals.append(sig)
        targets.append(np.reshape(
            dataset[k]['target'],
            (dataset[k]['target'].shape[0], 1)))

    groups = np.hstack(groups)
    X = np.vstack(signals)
    y = np.reshape(np.vstack(targets), (-1, 1))

    dataset = {'X': X, 'y': y, 'groups': groups}
    with open(data_dir / 'slimmed_dalia.pkl', 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

    return X, y, groups

def _get_data_gen_dalia_wesad(samples, targets, groups, data_dir, AUGMENT, dataset_name):
    subjects = 15 #number of patients on which PPG data is taken
    n = 4 #number of iteration in the same fold 
    
    indices, _ = _rndgroup_kfold(groups, n)
    kfold_it = 0
    while kfold_it < subjects:
        fold = kfold_it // n
        print(f'kFold-iteration: {kfold_it}')
        train_index, test_val_index = indices[fold]
        # Train Dataset
        train_samples = samples[train_index]
        train_targets = targets[train_index] #target = hr estimation
        if AUGMENT and dataset_name == "DALIA":
            print(f"=> Performing data augmentation. Please wait...")
            Augmenter = utils.Data_Augmentation(train_samples, train_targets, augmentations, data_dir)
            train_samples, train_targets = Augmenter.run()
        ds_train = Dalia(train_samples, train_targets)
        # Val and Test Dataset
        logo = LeaveOneGroupOut()
        samples_val_test = samples[test_val_index]
        targets_val_test = targets[test_val_index]
        groups_val_test = groups[test_val_index]
        j = 0
        for val_index, test_index in logo.split(samples_val_test,
                                                targets_val_test,
                                                groups_val_test):

            if j == kfold_it % n:
                val_samples = samples_val_test[val_index]
                val_targets = targets_val_test[val_index]
                ds_val = Dalia(val_samples, val_targets)
                test_subj = groups[test_val_index][test_index][0]
                print(f'Test Subject: {test_subj}')
                print(f"Test & Val Subjects: {np.unique(groups[test_val_index])}")
                test_samples = samples_val_test[test_index]
                test_targets = targets_val_test[test_index]
                
                ds_test = Dalia(test_samples, test_targets, test_subj) 
               #the dataset is unzipped in several 'Sx' files, each with a different number of samples 
               #for the different activities (that are the same for all the subjects)
               #we use k-fold to train the TempoNet several Sx during training and then test 
               #on a different and unseen Sx at each iteration
                
            j += 1

        yield ds_train, ds_val, ds_test
        kfold_it += 1


def _rndgroup_kfold(groups, n, seed=35):
    """
    Random analogous of sklearn.model_selection.GroupKFold.split.
    :return: list of (train, test) indices
    """
    groups = pd.Series(groups)
    ix = np.arange(len(groups))
    unique = np.unique(groups)
    np.random.RandomState(seed).shuffle(unique)
    indices = list()
    split_dict = dict()
    i = 0
    for split in np.array_split(unique, n):
        split_dict[i] = split
        i += 1
        mask = groups.isin(split)
        train, test = ix[~mask], ix[mask]
        indices.append((train, test))
    return indices, split_dict


class Dalia(Dataset):
    def __init__(self, samples, targets, test_subj=None):
        super(Dalia).__init__()
        self.samples = samples
        self.targets = targets
        self.test_subj = test_subj

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = torch.from_numpy(self.samples[idx]).float()
        sampleT = self._transformation(sample)
        target = self.targets[idx]
        return sampleT, target

    def _transformation(self, sample):
        sample = torch.tensor(np.expand_dims(sample, axis= -1))
        return sample

    def __len__(self):
        return len(self.samples)


def get_data(dataset_name = "WESAD",data_dir=None,url=WESAD_URL,ds_name='ppg_dalia.zip',kfold=True, augment = False):
    folder = ""
    if dataset_name == "WESAD":
      folder = "WESAD"
      url = WESAD_URL
    elif dataset_name == "DALIA":
      folder = "PPG_FieldStudy"
      url = DALIA_URL
    
    if data_dir is None:
        data_dir = Path('.').absolute() / dataset_name
        print(f"data dir = {data_dir}")
    filename = data_dir / ds_name
    # Download if does not exist
    if not filename.exists():
        print('Download in progress... Please wait.')
        ds_dalia = requests.get(url)
        data_dir.mkdir()
        with open(filename, 'wb') as f:
            f.write(ds_dalia.content)
    # Unzip if needed
    if not (data_dir / folder).exists():
        print('Unzip files... Please wait.')
        with zipfile.ZipFile(filename) as zf:
            zf.extractall(data_dir)

    # This step slims the dataset. This will help to speedup following usage of data
    if not (data_dir / 'slimmed_dalia.pkl').exists():
        dataset = _collect_data(data_dir, dataset_name)
        samples, target, groups = _preprocess_data(data_dir, dataset, dataset_name)
    else:
        with open(data_dir / 'slimmed_dalia.pkl', 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')
        samples, target, groups = dataset.values()
    
    generator = _get_data_gen_dalia_wesad(samples, target, groups, data_dir = data_dir, AUGMENT = augment, dataset_name = dataset_name)
    
    return generator 

def get_full_dataset(dataset_name,  data_dir=None, url=WESAD_URL, ds_name='ppg_dalia.zip'):
    folder = ""
    if dataset_name == "WESAD":
      folder = "WESAD"
      url = WESAD_URL
    elif dataset_name == "DALIA":
      folder = "PPG_FieldStudy"
      url = DALIA_URL
       
    if data_dir is None:
        data_dir = Path('.').absolute() / dataset_name
        print(f"data dir = {data_dir}")
    filename = data_dir / ds_name
    # Download if does not exist
    if not filename.exists():
        print('Download in progress... Please wait.')
        ds_dalia = requests.get(url)
        data_dir.mkdir()
        with open(filename, 'wb') as f:
            f.write(ds_dalia.content)
    # Unzip if needed
    if not (data_dir / folder).exists():
        print('Unzip files... Please wait.')
        with zipfile.ZipFile(filename) as zf:
            zf.extractall(data_dir)
                
    # This step slims the dataset. This will help to speedup following usage of data
    if not (data_dir / 'slimmed_dalia.pkl').exists():
        dataset = _collect_data(data_dir, dataset_name)
        samples, target, groups = _preprocess_data(data_dir, dataset, dataset_name)
    else:
        with open(data_dir / 'slimmed_dalia.pkl', 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')
        samples, target, groups = dataset.values()
        
    #Create full dataset
    full_dataset = Dalia(samples, target)
    train_dl = DataLoader(full_dataset,batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
    return train_dl
  

def build_dataloaders(datasets: Tuple[Dataset, ...],
                      batch_size=128,
                      num_workers=1
                      ):
    train_set, val_set, test_set = datasets
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True, #False
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


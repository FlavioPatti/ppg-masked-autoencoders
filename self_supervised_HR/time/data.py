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
import scipy.io
import copy
import pdb
import numpy as np

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
      num = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
    elif data == "DALIA":
      folder = "PPG_FieldStudy"
      num = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    elif data == "IEEETRAIN":
      num = [1,2,3,4,5,6,7,8,9,10,11,12]
      ty = [1,2,2,2,2,2,2,2,2,2,2,2]

    dataset = dict()
    session_list = random.sample(num, len(num))
    
    if data == "DALIA" or data == "WESAD":
      for subj in session_list:
        with open(data_dir / folder / f'S{str(subj)}' / f'S{str(subj)}.pkl', 'rb') as f:
            subject = pickle.load(f, encoding='latin1')
            ppg = subject['signal']['wrist']['BVP'][::2].astype('float32')
            acc = subject['signal']['wrist']['ACC'].astype('float32')
        if data == "DALIA":
            target = subject['label'].astype('float32')
        elif data == "WESAD":
            sub = 'S'+str(subj)
            filename = './ppg-masked-autoencoders/WESAD/WESAD'
            fs = 700
            
            #retrive ecg_signal from SX_respiban.txt
            ecg_signal = pd.read_csv(f'{filename}/{sub}/{sub}_respiban.txt', skiprows= 3, sep ='\t').iloc[:, 2].values
            
            #ECG R-peak detection
            _, results = neurokit2.ecg_peaks(ecg_signal, sampling_rate=fs)
            rpeaks = results["ECG_R_Peaks"]
            
            #Correct peaks
            #rpeaks_corrected = wfdb.processing.correct_peaks(
            #ecg_signal, rpeaks, search_radius=36, smooth_window_size=50, peak_dir="up")
            
            # Compute time intervals between consecutive peaks
            intervalli_tempo = [rpeaks[i+1] - rpeaks[i] for i in range(len(rpeaks)-1)]
            # Compute HR in BPM
            heart_rates = [60 / (intervallo_tempo / fs) for intervallo_tempo in intervalli_tempo]
            
            target = np.array(heart_rates).astype('float32')
        dataset[subj] = { 
        #each sample is build by: ppg value, accelerometer value, hr estimation
          'ppg': ppg,
          'acc': acc,
          'target': target
              }
    elif data == "IEEETRAIN":
      for idx, subj in enumerate (num):
        i = 0
        if subj <= 9:
          sub = '0'+str(subj)
          t = '0'+str(ty[idx])
        fs = 125
        data = scipy.io.loadmat(f'{data_dir}/DATA_{sub}_TYPE{t}.mat')['sig']
        #la prima riga = ECG signals
        ecg_signal = data[0:1, :]
        ecg_signal = np.squeeze(ecg_signal)
        #seconda e terza riga = PPG signals
        ppg = data[1:3, :]
        ppg = np.transpose(ppg, (1, 0))
        # ultime tre righe = acc signals
        acc = data[3:6, :]
        acc = np.transpose(acc, (1, 0))

        #hr = scipy.io.loadmat(f'{data_dir}/DATA_{sub}_TYPE{t}_BPMtrace.mat')['BPM0']
        #print(f"hr = {hr.shape}")
        
        fs = 125
        #ECG R-peak detection
        _, results = neurokit2.ecg_peaks(ecg_signal, sampling_rate=fs)
        rpeaks = results["ECG_R_Peaks"]
        #print(f"rpeaks = {rpeaks.shape}")
        
        #Correct peaks
        #rpeaks_corrected = wfdb.processing.correct_peaks(
        #ecg_signal, rpeaks, search_radius=36, smooth_window_size=50, peak_dir="up")

        # Compute time intervals between consecutive peaks
        intervalli_tempo = [rpeaks[i+1] - rpeaks[i] for i in range(len(rpeaks)-1)]
        #print(f"intervalli_tempo = {intervalli_tempo}")

        # Compute HR in BPM
        heart_rates = [60 / (intervallo_tempo / fs) for intervallo_tempo in intervalli_tempo]
        
        #target =  np.squeeze(hr).astype('float32')
        target = np.array(heart_rates).astype('float32')
        #print(f"target = {target.shape}")
    
        dataset[subj] = { 
        #each sample is build by: ppg value, accelerometer value, hr estimation
        'ppg': ppg,
        'acc': acc,
        'target': target
            }
    return dataset


def _preprocess_data(data_dir, dataset):
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

def _get_data_gen(samples, targets, groups, data_dir, AUGMENT):
    n = 4
    subjects = 15 #number of patients on which PPG data is taken
    indices, _ = _rndgroup_kfold(groups, n)
    kfold_it = 0
    while kfold_it < subjects:
        fold = kfold_it // n
        print(f'kFold-iteration: {kfold_it}')
        train_index, test_val_index = indices[fold]
        # Train Dataset
        train_samples = samples[train_index]
        train_targets = targets[train_index] #target = hr estimation
        if AUGMENT:
            print(f"=> Performing data augmentation. Please wait...")
            Augmenter = Data_Augmentation(train_samples, train_targets, augmentations, data_dir)
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


def get_data(dataset = "WESAD",data_dir=None,url=WESAD_URL,ds_name='ppg_dalia.zip',kfold=True, augment = False):
    folder = ""
    if dataset == "WESAD":
      folder = "WESAD"
      url = WESAD_URL
    elif dataset == "DALIA":
      folder = "PPG_FieldStudy"
      url = DALIA_URL
    
    if dataset == "DALIA" or dataset == "WESAD":
      if data_dir is None:
          data_dir = Path('.').absolute() / dataset
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
    if dataset == "IEEETRAIN":
        data_dir = Path('.').absolute() / 'IEEEPPG' / 'Training_data'
        # set data folder, train & test
        data_folder = "./IEEEPPG/"
        train_file = data_folder + "competition_data.zip"
        test_file = data_folder + "TestData.zip"
        with zipfile.ZipFile(train_file) as zf:
          zf.extractall(data_folder)
        with zipfile.ZipFile(test_file) as zf:
          zf.extractall(data_folder)

    # This step slims the dataset. This will help to speedup following usage of data
    if not (data_dir / 'slimmed_dalia.pkl').exists():
        dataset = _collect_data(data_dir, dataset)
        samples, target, groups = _preprocess_data(data_dir, dataset)
    else:
        with open(data_dir / 'slimmed_dalia.pkl', 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')
        samples, target, groups = dataset.values()
      
    generator = _get_data_gen(samples, target, groups, data_dir = data_dir, AUGMENT = augment)
    return generator 

def get_full_dataset(dataset,  data_dir=None, url=WESAD_URL, ds_name='ppg_dalia.zip'):
    folder = ""
    if dataset == "WESAD":
      folder = "WESAD"
      url = WESAD_URL
    elif dataset == "DALIA":
      folder = "PPG_FieldStudy"
      url = DALIA_URL
      
    if dataset == "DALIA" or dataset == "WESAD":  
        if data_dir is None:
            data_dir = Path('.').absolute() / dataset
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
    if dataset == "IEEETRAIN":
        data_dir = Path('.').absolute() / 'IEEEPPG' / 'Training_data'
        # set data folder, train & test
        data_folder = "./IEEEPPG/"
        train_file = data_folder + "competition_data.zip"
        test_file = data_folder + "TestData.zip"
        with zipfile.ZipFile(train_file) as zf:
          zf.extractall(data_folder)
        with zipfile.ZipFile(test_file) as zf:
          zf.extractall(data_folder)
                
    dataset = _collect_data(data_dir, dataset)
    samples, target, groups = _preprocess_data(data_dir, dataset)
    full_dataset = Dalia(samples, target)
    train_dl = DataLoader(full_dataset,batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
    return train_dl
  

def build_dataloaders(datasets: Tuple[Dataset, ...],
                      batch_size=128,
                      num_workers=4
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
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader

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

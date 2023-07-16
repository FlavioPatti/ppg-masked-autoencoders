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
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
    else:
      print('Try again, dataset must be only DALIA or WESAD')

    dataset = dict()
    session_list = random.sample(num, len(num))
    for subj in session_list:
        print(f"{data_dir}")
        with open(data_dir / folder / f'S{str(subj)}' / f'S{str(subj)}.pkl', 'rb') as f:
            subject = pickle.load(f, encoding='latin1')
        print(f"subject = {subject}")
        ppg = subject['signal']['wrist']['BVP'][::2].astype('float32')
        acc = subject['signal']['wrist']['ACC'].astype('float32')
        if data == "DALIA":
            target = subject['label'].astype('float32')
        elif data == "WESAD":
            sub = 'S'+str(subj)
            filename = '/content/ppg-masked-autoencoders/WESAD/WESAD'
            fs = 700
            
            #retrive ecg_signal from SX_respiban.txt
            ecg_signal = pd.read_csv(f'{filename}/{sub}/{sub}_respiban.txt', skiprows= 3, sep ='\t').iloc[:, 2].values
            print(f"ecg_signal = {ecg_signal}")
            
            #ECG R-peak detection
            _, results = neurokit2.ecg_peaks(ecg_signal, sampling_rate=fs)
            rpeaks = results["ECG_R_Peaks"]
            print(f"rpeaks = {rpeaks}")
            
            #Correct peaks
            rpeaks_corrected = wfdb.processing.correct_peaks(
            ecg_signal, rpeaks, search_radius=36, smooth_window_size=50, peak_dir="up")
            print(f"rpeaks = {rpeaks_corrected}")
            
            # Compute time intervals between consecutive peaks
            intervalli_tempo = [rpeaks_corrected[i+1] - rpeaks_corrected[i] for i in range(len(rpeaks)-1)]
            # Compute HR in BPM
            heart_rates = [60 / (intervallo_tempo / fs) for intervallo_tempo in intervalli_tempo]
            
            print(f"hr = {heart_rates}")
            target = np.array(heart_rates).astype('float32')
    
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

def _get_data_gen(samples, targets, groups):
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
        sample = self.samples[idx]
        sample = sample - np.mean(sample)

        target = self.targets[idx]
        
        return sample, target

    def __len__(self):
        return len(self.samples)


def get_data(dataset = "WESAD",
             data_dir=None,
             url=WESAD_URL,
             ds_name='ppg_dalia.zip',
             kfold=True):
    folder = ""
    if dataset == "WESAD":
      folder = "WESAD"
      url = WESAD_URL
    elif dataset == "DALIA":
      folder = "PPG_FieldStudy"
      url = DALIA_URL
    else:
      print('Try again, dataset must be only DALIA or WESAD')
      
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

    # This step slims the dataset. This will help to speedup following usage of data
    if not (data_dir / 'slimmed_dalia.pkl').exists():
        dataset = _collect_data(data_dir, dataset)
        samples, target, groups = _preprocess_data(data_dir, dataset)
    else:
        with open(data_dir / 'slimmed_dalia.pkl', 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')
        samples, target, groups = dataset.values()
      
    generator = _get_data_gen(samples, target, groups)
    return generator 

def get_full_dataset(dataset,  data_dir=None, url=WESAD_URL, ds_name='ppg_dalia.zip'):
    folder = ""
    if dataset == "WESAD":
      folder = "WESAD"
      url = WESAD_URL
    elif dataset == "DALIA":
      folder = "PPG_FieldStudy"
      url = DALIA_URL
    else:
      print('Try again, dataset must be only DALIA or WESAD')
      
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
                
        # This step slims the dataset. This will help to speedup following usage of data
        if not (data_dir / 'slimmed_dalia.pkl').exists():
            dataset = _collect_data(data_dir, dataset)
            samples, target, groups = _preprocess_data(data_dir, dataset)
        else:
            with open(data_dir / 'slimmed_dalia.pkl', 'rb') as f:
                dataset = pickle.load(f, encoding='latin1')
            samples, target, groups = dataset.values()
            
        full_dataset = Dalia(samples, target)
        train_dl = DataLoader(full_dataset,batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
        return train_dl
    
    else: #IEEEPPG 
        # set data folder, train & test
        data_folder = "./IEEEPPG/"
        train_file = data_folder + "IEEEPPG_TRAIN.ts"
        test_file = data_folder + "IEEEPPG_TEST.ts"
        norm = "none"
        
        # loading the data. X_train and X_test are dataframe of N x n_dim
        X_train, y_train = load_from_tsfile_to_dataframe(train_file)
        X_test, y_test = load_from_tsfile_to_dataframe(test_file)

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
        x_train = process_data(X_train, normalise=norm, min_len=min_len)
        x_train = np.transpose(x_train, (0, 2, 1)).astype(np.double)
        x_test = process_data(X_test, normalise=norm, min_len=min_len)
        x_test = np.transpose(x_test, (0, 2, 1)).astype(np.double)
        
        #retrive training and validation from training data
        #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        #retrive dataloaders
        full_dataset = Dalia(x_train, y_train)
        train_dl = DataLoader(full_dataset,batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
        return train_dl        


def uniform_scaling(data, max_len):
    """
    This is a function to scale the time series uniformly
    :param data:
    :param max_len:
    :return:
    """
    seq_len = len(data)
    scaled_data = [data[int(j * seq_len / max_len)] for j in range(max_len)]

    return scaled_data

name = "DataLoader"

regression_datasets = ["AustraliaRainfall",
                       "HouseholdPowerConsumption1",
                       "HouseholdPowerConsumption2",
                       "BeijingPM25Quality",
                       "BeijingPM10Quality",
                       "Covid3Month",
                       "LiveFuelMoistureContent",
                       "FloodModeling1",
                       "FloodModeling2",
                       "FloodModeling3",
                       "AppliancesEnergy",
                       "BenzeneConcentration",
                       "NewsHeadlineSentiment",
                       "NewsTitleSentiment",
                       "BIDMC32RR",
                       "BIDMC32HR",
                       "BIDMC32SpO2",
                       "IEEEPPG",
                       "PPGDalia"]


# The following code is adapted from the python package sktime to read .ts file.
class TsFileParseException(Exception):
    """
    Should be raised when parsing a .ts file and the format is incorrect.
    """
    pass


def load_from_tsfile_to_dataframe(full_file_path_and_name, return_separate_X_and_y=True,
                                  replace_missing_vals_with='NaN'):
    """Loads data from a .ts file into a Pandas DataFrame.

    Parameters
    ----------
    full_file_path_and_name: str
        The full pathname of the .ts file to read.
    return_separate_X_and_y: bool
        true if X and Y values should be returned as separate Data Frames (X) and a numpy array (y), false otherwise.
        This is only relevant for data that
    replace_missing_vals_with: str
       The value that missing values in the text file should be replaced with prior to parsing.

    Returns
    -------
    DataFrame, ndarray
        If return_separate_X_and_y then a tuple containing a DataFrame and a numpy array containing the relevant time-series and corresponding class values.
    DataFrame
        If not return_separate_X_and_y then a single DataFrame containing all time-series and (if relevant) a column "class_vals" the associated class values.
    """

    # Initialize flags and variables used when parsing the file
    metadata_started = False
    data_started = False

    has_problem_name_tag = False
    has_timestamps_tag = False
    has_univariate_tag = False
    has_class_labels_tag = False
    has_target_labels_tag = False
    has_data_tag = False

    previous_timestamp_was_float = None
    previous_timestamp_was_int = None
    previous_timestamp_was_timestamp = None
    num_dimensions = None
    is_first_case = True
    instance_list = []
    class_val_list = []
    line_num = 0

    # Parse the file
    # print(full_file_path_and_name)
    with open(full_file_path_and_name, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            # print(".", end='')
            # Strip white space from start/end of line and change to lowercase for use below
            line = line.strip().lower()
            # Empty lines are valid at any point in a file
            if line:
                # Check if this line contains metadata
                # Please note that even though metadata is stored in this function it is not currently published externally
                if line.startswith("@problemname"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")
                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException("problemname tag requires an associated value")

                    problem_name = line[len("@problemname") + 1:]
                    has_problem_name_tag = True
                    metadata_started = True
                elif line.startswith("@timestamps"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)

                    if token_len != 2:
                        raise TsFileParseException("timestamps tag requires an associated Boolean value")
                    elif tokens[1] == "true":
                        timestamps = True
                    elif tokens[1] == "false":
                        timestamps = False
                    else:
                        raise TsFileParseException("invalid timestamps value")
                    has_timestamps_tag = True
                    metadata_started = True
                elif line.startswith("@univariate"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)
                    if token_len != 2:
                        raise TsFileParseException("univariate tag requires an associated Boolean value")
                    elif tokens[1] == "true":
                        univariate = True
                    elif tokens[1] == "false":
                        univariate = False
                    else:
                        raise TsFileParseException("invalid univariate value")

                    has_univariate_tag = True
                    metadata_started = True
                elif line.startswith("@classlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException("classlabel tag requires an associated Boolean value")

                    if tokens[1] == "true":
                        class_labels = True
                    elif tokens[1] == "false":
                        class_labels = False
                    else:
                        raise TsFileParseException("invalid classLabel value")

                    # Check if we have any associated class values
                    if token_len == 2 and class_labels:
                        raise TsFileParseException("if the classlabel tag is true then class values must be supplied")

                    has_class_labels_tag = True
                    class_label_list = [token.strip() for token in tokens[2:]]
                    metadata_started = True
                elif line.startswith("@targetlabel"):
                    # Check that the data has not started
                    if data_started:
                        raise TsFileParseException("metadata must come before data")

                    # Check that the associated value is valid
                    tokens = line.split(' ')
                    token_len = len(tokens)

                    if token_len == 1:
                        raise TsFileParseException("targetlabel tag requires an associated Boolean value")

                    if tokens[1] == "true":
                        target_labels = True
                    elif tokens[1] == "false":
                        target_labels = False
                    else:
                        raise TsFileParseException("invalid targetLabel value")

                    has_target_labels_tag = True
                    class_val_list = []
                    metadata_started = True
                # Check if this line contains the start of data
                elif line.startswith("@data"):
                    if line != "@data":
                        raise TsFileParseException("data tag should not have an associated value")

                    if data_started and not metadata_started:
                        raise TsFileParseException("metadata must come before data")
                    else:
                        has_data_tag = True
                        data_started = True
                # If the 'data tag has been found then metadata has been parsed and data can be loaded
                elif data_started:
                    # Check that a full set of metadata has been provided
                    incomplete_regression_meta_data = not has_problem_name_tag or not has_timestamps_tag or not has_univariate_tag or not has_target_labels_tag or not has_data_tag
                    incomplete_classification_meta_data = not has_problem_name_tag or not has_timestamps_tag or not has_univariate_tag or not has_class_labels_tag or not has_data_tag
                    if incomplete_regression_meta_data and incomplete_classification_meta_data:
                        raise TsFileParseException("a full set of metadata has not been provided before the data")

                    # Replace any missing values with the value specified
                    line = line.replace("?", replace_missing_vals_with)

                    # Check if we dealing with data that has timestamps
                    if timestamps:
                        # We're dealing with timestamps so cannot just split line on ':' as timestamps may contain one
                        has_another_value = False
                        has_another_dimension = False

                        timestamps_for_dimension = []
                        values_for_dimension = []

                        this_line_num_dimensions = 0
                        line_len = len(line)
                        char_num = 0

                        while char_num < line_len:
                            # Move through any spaces
                            while char_num < line_len and str.isspace(line[char_num]):
                                char_num += 1

                            # See if there is any more data to read in or if we should validate that read thus far

                            if char_num < line_len:

                                # See if we have an empty dimension (i.e. no values)
                                if line[char_num] == ":":
                                    if len(instance_list) < (this_line_num_dimensions + 1):
                                        instance_list.append([])

                                    instance_list[this_line_num_dimensions].append(pd.Series())
                                    this_line_num_dimensions += 1

                                    has_another_value = False
                                    has_another_dimension = True

                                    timestamps_for_dimension = []
                                    values_for_dimension = []

                                    char_num += 1
                                else:
                                    # Check if we have reached a class label
                                    if line[char_num] != "(" and target_labels:
                                        class_val = line[char_num:].strip()

                                        # if class_val not in class_val_list:
                                        #     raise TsFileParseException(
                                        #         "the class value '" + class_val + "' on line " + str(
                                        #             line_num + 1) + " is not valid")

                                        class_val_list.append(float(class_val))
                                        char_num = line_len

                                        has_another_value = False
                                        has_another_dimension = False

                                        timestamps_for_dimension = []
                                        values_for_dimension = []

                                    else:

                                        # Read in the data contained within the next tuple

                                        if line[char_num] != "(" and not target_labels:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " does not start with a '('")

                                        char_num += 1
                                        tuple_data = ""

                                        while char_num < line_len and line[char_num] != ")":
                                            tuple_data += line[char_num]
                                            char_num += 1

                                        if char_num >= line_len or line[char_num] != ")":
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " does not end with a ')'")

                                        # Read in any spaces immediately after the current tuple

                                        char_num += 1

                                        while char_num < line_len and str.isspace(line[char_num]):
                                            char_num += 1

                                        # Check if there is another value or dimension to process after this tuple

                                        if char_num >= line_len:
                                            has_another_value = False
                                            has_another_dimension = False

                                        elif line[char_num] == ",":
                                            has_another_value = True
                                            has_another_dimension = False

                                        elif line[char_num] == ":":
                                            has_another_value = False
                                            has_another_dimension = True

                                        char_num += 1

                                        # Get the numeric value for the tuple by reading from the end of the tuple data backwards to the last comma

                                        last_comma_index = tuple_data.rfind(',')

                                        if last_comma_index == -1:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains a tuple that has no comma inside of it")

                                        try:
                                            value = tuple_data[last_comma_index + 1:]
                                            value = float(value)

                                        except ValueError:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains a tuple that does not have a valid numeric value")

                                        # Check the type of timestamp that we have

                                        timestamp = tuple_data[0: last_comma_index]

                                        try:
                                            timestamp = int(timestamp)
                                            timestamp_is_int = True
                                            timestamp_is_timestamp = False
                                        except ValueError:
                                            timestamp_is_int = False

                                        if not timestamp_is_int:
                                            try:
                                                timestamp = float(timestamp)
                                                timestamp_is_float = True
                                                timestamp_is_timestamp = False
                                            except ValueError:
                                                timestamp_is_float = False

                                        if not timestamp_is_int and not timestamp_is_float:
                                            try:
                                                timestamp = timestamp.strip()
                                                timestamp_is_timestamp = True
                                            except ValueError:
                                                timestamp_is_timestamp = False

                                        # Make sure that the timestamps in the file (not just this dimension or case) are consistent

                                        if not timestamp_is_timestamp and not timestamp_is_int and not timestamp_is_float:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains a tuple that has an invalid timestamp '" + timestamp + "'")

                                        if previous_timestamp_was_float is not None and previous_timestamp_was_float and not timestamp_is_float:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                        if previous_timestamp_was_int is not None and previous_timestamp_was_int and not timestamp_is_int:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                        if previous_timestamp_was_timestamp is not None and previous_timestamp_was_timestamp and not timestamp_is_timestamp:
                                            raise TsFileParseException(
                                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                                    line_num + 1) + " contains tuples where the timestamp format is inconsistent")

                                        # Store the values

                                        timestamps_for_dimension += [timestamp]
                                        values_for_dimension += [value]

                                        #  If this was our first tuple then we store the type of timestamp we had

                                        if previous_timestamp_was_timestamp is None and timestamp_is_timestamp:
                                            previous_timestamp_was_timestamp = True
                                            previous_timestamp_was_int = False
                                            previous_timestamp_was_float = False

                                        if previous_timestamp_was_int is None and timestamp_is_int:
                                            previous_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = True
                                            previous_timestamp_was_float = False

                                        if previous_timestamp_was_float is None and timestamp_is_float:
                                            previous_timestamp_was_timestamp = False
                                            previous_timestamp_was_int = False
                                            previous_timestamp_was_float = True

                                        # See if we should add the data for this dimension

                                        if not has_another_value:
                                            if len(instance_list) < (this_line_num_dimensions + 1):
                                                instance_list.append([])

                                            if timestamp_is_timestamp:
                                                timestamps_for_dimension = pd.DatetimeIndex(timestamps_for_dimension)

                                            instance_list[this_line_num_dimensions].append(
                                                pd.Series(index=timestamps_for_dimension, data=values_for_dimension))
                                            this_line_num_dimensions += 1

                                            timestamps_for_dimension = []
                                            values_for_dimension = []

                            elif has_another_value:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                        line_num + 1) + " ends with a ',' that is not followed by another tuple")

                            elif has_another_dimension and target_labels:
                                raise TsFileParseException(
                                    "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                        line_num + 1) + " ends with a ':' while it should list a class value")

                            elif has_another_dimension and not target_labels:
                                if len(instance_list) < (this_line_num_dimensions + 1):
                                    instance_list.append([])

                                instance_list[this_line_num_dimensions].append(pd.Series(dtype=np.float32))
                                this_line_num_dimensions += 1
                                num_dimensions = this_line_num_dimensions

                            # If this is the 1st line of data we have seen then note the dimensions

                            if not has_another_value and not has_another_dimension:
                                if num_dimensions is None:
                                    num_dimensions = this_line_num_dimensions

                                if num_dimensions != this_line_num_dimensions:
                                    raise TsFileParseException("line " + str(
                                        line_num + 1) + " does not have the same number of dimensions as the previous line of data")

                        # Check that we are not expecting some more data, and if not, store that processed above

                        if has_another_value:
                            raise TsFileParseException(
                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                    line_num + 1) + " ends with a ',' that is not followed by another tuple")

                        elif has_another_dimension and target_labels:
                            raise TsFileParseException(
                                "dimension " + str(this_line_num_dimensions + 1) + " on line " + str(
                                    line_num + 1) + " ends with a ':' while it should list a class value")

                        elif has_another_dimension and not target_labels:
                            if len(instance_list) < (this_line_num_dimensions + 1):
                                instance_list.append([])

                            instance_list[this_line_num_dimensions].append(pd.Series())
                            this_line_num_dimensions += 1
                            num_dimensions = this_line_num_dimensions

                        # If this is the 1st line of data we have seen then note the dimensions

                        if not has_another_value and num_dimensions != this_line_num_dimensions:
                            raise TsFileParseException("line " + str(
                                line_num + 1) + " does not have the same number of dimensions as the previous line of data")

                        # Check if we should have class values, and if so that they are contained in those listed in the metadata

                        if target_labels and len(class_val_list) == 0:
                            raise TsFileParseException("the cases have no associated class values")
                    else:
                        dimensions = line.split(":")
                        # If first row then note the number of dimensions (that must be the same for all cases)
                        if is_first_case:
                            num_dimensions = len(dimensions)

                            if target_labels:
                                num_dimensions -= 1

                            for dim in range(0, num_dimensions):
                                instance_list.append([])
                            is_first_case = False

                        # See how many dimensions that the case whose data in represented in this line has
                        this_line_num_dimensions = len(dimensions)

                        if target_labels:
                            this_line_num_dimensions -= 1

                        # All dimensions should be included for all series, even if they are empty
                        if this_line_num_dimensions != num_dimensions:
                            raise TsFileParseException("inconsistent number of dimensions. Expecting " + str(
                                num_dimensions) + " but have read " + str(this_line_num_dimensions))

                        # Process the data for each dimension
                        for dim in range(0, num_dimensions):
                            dimension = dimensions[dim].strip()

                            if dimension:
                                data_series = dimension.split(",")
                                data_series = [float(i) for i in data_series]
                                instance_list[dim].append(pd.Series(data_series))
                            else:
                                instance_list[dim].append(pd.Series())

                        if target_labels:
                            class_val_list.append(float(dimensions[num_dimensions].strip()))

            line_num += 1

    # Check that the file was not empty
    if line_num:
        # Check that the file contained both metadata and data
        complete_regression_meta_data = has_problem_name_tag and has_timestamps_tag and has_univariate_tag and has_target_labels_tag and has_data_tag
        complete_classification_meta_data = has_problem_name_tag and has_timestamps_tag and has_univariate_tag and has_class_labels_tag and has_data_tag

        if metadata_started and not complete_regression_meta_data and not complete_classification_meta_data:
            raise TsFileParseException("metadata incomplete")
        elif metadata_started and not data_started:
            raise TsFileParseException("file contained metadata but no data")
        elif metadata_started and data_started and len(instance_list) == 0:
            raise TsFileParseException("file contained metadata but no data")

        # Create a DataFrame from the data parsed above
        data = pd.DataFrame(dtype=np.float32)

        for dim in range(0, num_dimensions):
            data['dim_' + str(dim)] = instance_list[dim]

        # Check if we should return any associated class labels separately

        if target_labels:
            if return_separate_X_and_y:
                return data, np.asarray(class_val_list)
            else:
                data['class_vals'] = pd.Series(class_val_list)
                return data
        else:
            return data
    else:
        raise TsFileParseException("empty file")

def process_data(X, min_len, normalise=None):
    """
    This is a function to process the data, i.e. convert dataframe to numpy array
    :param X:
    :param min_len:
    :param normalise:
    :return:
    """
    tmp = []
    for i in tqdm(range(len(X))):
        _x = X.iloc[i, :].copy(deep=True)

        # 1. find the maximum length of each dimension
        all_len = [len(y) for y in _x]
        max_len = max(all_len)

        # 2. adjust the length of each dimension
        _y = []
        for y in _x:
            # 2.1 fill missing values
            if y.isnull().any():
                y = y.interpolate(method='linear', limit_direction='both')

            # 2.2. if length of each dimension is different, uniformly scale the shorted one to the max length
            if len(y) < max_len:
                y = uniform_scaling(y, max_len)
            _y.append(y)
        _y = np.array(np.transpose(_y))

        # 3. adjust the length of the series, chop of the longer series
        _y = _y[:min_len, :]

        # 4. normalise the series
        if normalise == "standard":
            scaler = StandardScaler().fit(_y)
            _y = scaler.transform(_y)
        if normalise == "minmax":
            scaler = MinMaxScaler().fit(_y)
            _y = scaler.transform(_y)

        tmp.append(_y)
    X = np.array(tmp)
    return X

def get_dataloaders(x_train, x_val, x_test, y_train, y_val, y_test):
        
    train_dataset = Dalia(x_train, y_train)
    val_dataset = Dalia(x_val,y_val)
    test_dataset = Dalia(x_test, y_test)
    
    train_dl = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        num_workers=4)
    val_dl = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        num_workers=4)
    test_dl = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        num_workers=4)
    
    return train_dl, val_dl, test_dl
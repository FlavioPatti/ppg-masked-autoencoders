# Heart Rate Detection with Masked AutoEncoders

The repository is entirely written in pytorch and addresses the training of Masked Autoencoder models for **heart rate estimation** based on wrist wearable devices.

## Installation
To install the latest release:

```
$ git clone https://github.com/eml-eda/ppg-masked-autoencoders.git
$ cd ppg-masked-autoencoders
$ python setup.py install
```

## How to setup the environment 
To run the experiments read and install all the libraries present in `requirements.txt`

## API Details
In this repository you can find two types of experiments: signal reconstruction (pre-train phase) + heart rate estimation (finetuning phase) in the time domain and the equivalent in the frequency domain. To run these two experiments the corrisponding files are [`launch_time_experiments.py`](#launch_time_experimentspy) and [`launch_frequency_experiments.py`](#launch_frequency_experimentspy).

For each experiment two datasets can be tested: **PPG_Dalia** and **WESAD**. If you want to apply transfer learning step, please, set *TRANSFER_LEARNING = True* and specify the name of the datasets for the pre-train (*DATASET_PRETRAIN*) and finetuning (*DATASET_FINETUNING*) phases, otherwise the experiment is executed with the same dataset following a k-fold cross validation approach.
Each experiment is a stand-alone python module based on five python files, namely:
1. [`data.py`](#datapy)
2. [`model_pretrain.py`](#model_pretrainpy)
3. [`model_finetune.py`](#model_finetunepy)
4. [`train_time/freq.py`](#train_time/freqpy)
5. [`__init__.py`](#__init__py)

You can find these files in their corrisponding folders: `self_supervised_HR/time` and `self_supervised_HR/frequency`.

#### **`data.py`**
This module implement all the functions needed to gather the data, pre-process them and finally ship them to the user both in the form of [Pytorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) and [Pytorch Dataloader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). 

The main function in this module are:
- `get_data`, which returns a tuple of [Pytorch Datasets](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). The number of returned datasets is 3 (train, validation and test). The argument of the function is directly the link to download the entire data zip in the cases of PPG_Dalia and WESAD. For IEEE_Train & Test, instead, since these datasets are much smaller, you can directly find them in this repository.

- `build_dataloaders`, which returns a tuple of [Pytorch Dataloaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). Takes as inputs the dataset returned by `get_data` and constants such as the *batch-size* and the *number of workers*. The number of elements of the returned tuple is 3 (train,validation, test) according to `get_data` implementation.

#### **`model_pretrain.py`**
This module implement the **MaskedAutoencoderViT** (Masked Autoencoder Vision Trasformer) to reconstruct the input signal from the original version applying a certain *mask_ratio* after *N_PRETRAIN_EPOCHS* epochs. The optimizer used is **AdamW** and the criterion is **MSE Loss**.

The input is the audio of the PPG signal in time experiment and the corrisponding spectogram in frequency experiment. 

The actual configurations use a `patch_size = (1,1)` for time and a `patch_size= (8,8)` for frequency in order to reach a uniform number of patches in the two experiments of 256.

#### **`model_finetune.py`**
This module implements an architecture similar to the previous one but without the decoder in the Masked Autoencoder ViT. 

In its place a regression tail was inserted consisting of two Convolutional layers a AvgPooling and a final Linear layer, which starting from the extracted features of the pre-training step try to use this acquired knowledge to predict the Heart Rate of the various patients. 

A Masked Autoencoder-big was choosen for the time experiments are: 
- `depth` = 12, 
- `heads` = 16, 
- `embed_dim`= 256

A Masked Autoencoder-small was choosen for the frequency experiments are: 
- `depth` = 4, 
- `heads` = 16, 
- `embed_dim`= 64

but the model has been made flexible to adapt to any correct changes of these parameters.

Futhermore, the model is trained for *DATASET_FINETUNING* epochs with an early stop of 20 epochs on the validation MAE. The optimizer used is **Adam** and the criterion is **LogCosh**.

All the results for the various experiments conducted in this repository are presented in *results.xlsx*

#### **`train_time/freq.py`**
This module implement the minimum set of information required to implement a training loop.

- `train_one_epoch_masked_autoencoder`, implements one epoch of training for to reconstruct the input signal. It takes as input an integer specifying the current *epoch*, the *model* to be trained, the *criterion*, the *optimizer*, the *train* and *val* dataloaders and finally the *device* to be used for the training. It returns a dictionary of tracked metrics.
- `train_one_epoch_hr_detection`, implements one epoch of training and validation for predict heart rate estimation. For the validation part it directly calls the `evaluate` function. It takes as input an integer specifying the current *epoch*, the *model* to be trained, the *criterion*, the *optimizer*, the *train* and *val* dataloaders and finally the *device* to be used for the training. It returns a dictionary of tracked metrics.
- `evaluate`, implement an evaluation step of the model. This step can be both of validation or test depending on the specific dataloader provided as input. It takes as input the *model*, the *criterion*, the *dataloader* and the *device*. It returns a dictionary of tracked metrics.

#### **`__init__.py`**
The body of this file import all the standard functions described in `data.py`, `model_pretrain.py`, `model_finetune.py` and `train_time/freq.py`.
This file is mandatory to identify the parent directory as a python package and to expose to the user the developed functions.

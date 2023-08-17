# Heart Rate Detection with Masked AutoEncoders

The repository is entirely written in pytorch and addresses the training of DNNs models for **heart rate estimation** based on wrist wearable devices.

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
In this repository you can find two types of experiments: signal reconstruction (pre-train phase) + heart rate estimation (finetuning phase) in the time domain and the equivalent in the frequency domain. To run this two experiments the corrisponding files are [`launch_time_experiment.py`](#launch_time_experimentpy) and [`launch_freq_experiment.py`](#launch_freq_experimentpy).
Each experiment is a stand-alone python module based on five python files, namely:
1. [`data.py`](#datapy)
2. [`model_pretrain.py`](#model_pretrainpy)
3. [`model_finetune.py`](#model_finetunepy)
4. [`train.py`](#trainpy)
5. [`__init__.py`](#__init__py)

#### **`data.py`**
This module implement all the functions needed to gather the data, pre-process them and finally ship them to the user both in the form of [Pytorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) and [Pytorch Dataloader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). 

The dataset is **PPG-DaLiA** which contains data from 15 subjects wearing physiological and motion sensors, providing data for motion compensation and heart rate estimation in Daily Life Activities.

The main function in this module are:
- `get_data`, which returns a tuple of [Pytorch Datasets](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). The number of returned datasets is 3 (train, validation and test). The argument of the function is directly the [link](https://archive.ics.uci.edu/ml/machine-learning-databases/00495/data.zip) to download the entire data zip.

- `build_dataloaders`, which returns a tuple of [Pytorch Dataloaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). Takes as inputs the dataset returned by `get_data` and constants such as the *batch-size* and the *number of workers*. The number of elements of the returned tuple is 3 (train,validation, test) according to `get_data` implementation.

#### **`model_pretrain.py`**
This module implement the **MaskedAutoencoderViT** (Masked Autoencoder Vision Trasformer) to reconstruct the input signal after **100 epochs**. The optimizer used is **AdamW** and the criterion is **MSE Loss**.

The input is the audio of the PPG signal in time experiment and the corrisponding spectogram in frequency experiment. 

The actual configurations use a `patch_size` = (1,1) for time and a `patch_size` = (8,8) for frequency in order to reach a uniform number of patches in the two experiments of `256`.

You can see an example of reconstruction for both cases below where the first image is the input and the second image is the reconstruction.

## Time
![input_time](https://github.com/eml-eda/ppg-masked-autoencoders/assets/101011113/16fbd6f9-a223-4105-b8d3-65459da90b36)
![output_time](https://github.com/eml-eda/ppg-masked-autoencoders/assets/101011113/b47a91c1-b082-4a55-b6f6-458c61b38d1d)

## Frequence
![input_freq](https://github.com/eml-eda/ppg-masked-autoencoders/assets/101011113/796c2bb3-9cf5-4bb1-9001-611d0c569999)
![output_freq](https://github.com/eml-eda/ppg-masked-autoencoders/assets/101011113/daac96ab-f643-4eb2-9cf5-45fb0a5a94c9)

#### **`model_finetune.py`**
This module implements an architecture similar to the previous one but without the decoder in the Masked Autoencoder ViT. 

In its place two convolutional layers and a final linear layer have been inserted for the prediction of the Heart Rate of the 15 patients. 

The currently setting of the parameters are: 
- `depth` = 12, 
- `heads` = 16, 
- `embed_dim`= 256 

but the model has been made flexible to adapt to any changes of these parameters.

Futhermore, the model is trained for **200 epochs** with an early stop of 20 epochs over the validation MAE. The optimizer used is **Adam** and the criterion is **LogCosh**.

Actual results are presented here:
[results.xlsx](https://github.com/eml-eda/ppg-masked-autoencoders/files/11721189/results.xlsx)


#### **`train.py`**
This module implement the minimum set of information required to implement a training loop.

- `train_one_epoch_masked_autoencoder`, implements one epoch of training for to reconstruct the input signal. It takes as input an integer specifying the current *epoch*, the *model* to be trained, the *criterion*, the *optimizer*, the *train* and *val* dataloaders and finally the *device* to be used for the training. It returns a dictionary of tracked metrics.
- `train_one_epoch_hr_detection`, implements one epoch of training and validation for predict heart rate estimation. For the validation part it directly calls the `evaluate` function. It takes as input an integer specifying the current *epoch*, the *model* to be trained, the *criterion*, the *optimizer*, the *train* and *val* dataloaders and finally the *device* to be used for the training. It returns a dictionary of tracked metrics.
- `evaluate`, implement an evaluation step of the model. This step can be both of validation or test depending on the specific dataloader provided as input. It takes as input the *model*, the *criterion*, the *dataloader* and the *device*. It returns a dictionary of tracked metrics.

#### **`__init__.py`**
The body of this file import all the standard functions described in `data.py`, `model_pretrain.py`, `model_finetune.py` and `train.py`.
This file is mandatory to identify the parent directory as a python package and to expose to the user the developed functions.

### Example Scripts
Finally, an example script is provided that shows how to use the different functions in order to build a neat and simple DNN training:
run `lanch_time_experiment.py` to execute time experiment or `launch_freq_experiment` to execute frequence experiment.

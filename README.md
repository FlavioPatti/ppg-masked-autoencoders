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
This repository  is a stand-alone python module based on five python files, namely:
1. [`data.py`](#datapy)
2. [`model.py`](#modelpy)
3. [`train.py`](#trainpy)
4. [`__init__.py`](#__init__py)

#### **`data.py`**
This module **must** implement all the functions needed to gather the data, eventually pre-process them and finally ship them to the user both in the form of [Pytorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) and [Pytorch Dataloader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). 

The dataset is **PPG-DaLiA** which contains data from 15 subjects wearing physiological and motion sensors, providing data for motion compensation and heart rate estimation in Daily Life Activities.



The two mandatory and standard functions that need to be implemented are:
- `get_data`, which returns a tuple of [Pytorch Datasets](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). The number of returned datasets is 3 (train, validation and test). The argument of the function is directly the [link](https://archive.ics.uci.edu/ml/machine-learning-databases/00495/data.zip) to download the entire data zip.

- `build_dataloaders`, which returns a tuple of [Pytorch Dataloaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). Takes as inputs the dataset returned by `get_data` and constants such as the *batch-size* and the *number of workers*. The number of elements of the returned tuple is 3 (train,validation, test) according to `get_data` implementation.

#### **`model.py`**
This module **must** implement at least one model.

The mandatory and standard function that needs to be implemented is:
- `get_reference_model`, the function always take as first argument the *model_name* which is a string associated to a specific pytorch model. Optionally, the function can take as argument *model_config* i.e., a python dictionary of additional configurations for the model. It returns the requested pytorch model.

In this case, we first use a **ViT** (Vision Transformer) to reconstruct the input signal (the implementation of it is in `models_mae` file) and then a **TempoNet** to predict the heart rate estimation from the reconstructed signal.

If the provided *model_name* is not supported an error is raised.

#### **`train.py`**
This module **must** implement the minimum set of information required to implement a training loop.

In particular, the mandatory and standard functions that needs to be implemented are:
- `get_default_optimizer`, it takes as input the pytorch model returned by `get_reference_model` and returns a default optimizer, that in this case is the **Adam** algorithm. 
- `get_default_criterion`, it takes no inputs and returns a default loss function, that is this case is the **LogCosh** which computes the logarithm of the hyperbolic cosine of the prediction error.
- `train_one_epoch_masked_autoencoder`, implements one epoch of training and validation for to reconstruct the input signal. For the validation part it directly calls the `evaluate` function. It takes as input an integer specifying the current *epoch*, the *model* to be trained, the *criterion*, the *optimizer*, the *train* and *val* dataloaders and finally the *device* to be used for the training. It returns a dictionary of tracked metrics.
- `train_one_epoch_hr_detection`, implements one epoch of training and validation for predict heart rate estimation. For the validation part it directly calls the `evaluate` function. It takes as input an integer specifying the current *epoch*, the *model* to be trained, the *criterion*, the *optimizer*, the *train* and *val* dataloaders and finally the *device* to be used for the training. It returns a dictionary of tracked metrics.
- `evaluate`, implement an evaluation step of the model. This step can be both of validation or test depending on the specific dataloader provided as input. It takes as input the *model*, the *criterion*, the *dataloader* and the *device*. It returns a dictionary of tracked metrics.

Optionally, the benchmark may defines and implements the `get_default_scheduler` function which takes as input the optimizer and returns a specified learning-rate scheduler. 
At this point, no implementation of it is provided.

#### **`__init__.py`**
The body of this file **must** import all the standard functions described in `data.py`, `model.py` and `train.py`.
This file is mandatory to identify the parent directory as a python package and to expose to the user the developed functions.

To gain more insights about how this file is structurated and about how the user can develop one on its own, please consult one of the different `__init__.py` files already included in the library. E.g., [`image_classification/__init__.py`](./pytorch_benchmarks/image_classification/__init__.py).

### Example Scripts
Finally, an example script is provided that shows how to use the different functions in order to build a neat and simple DNN training:
[Heart Rate Detection Example](hr_detection_example.py)

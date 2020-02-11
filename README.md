# Image-Classifier
Training an Image Classifier in PyTorch framework by using Transfer Learning with Pre-Trained CNN Model Architectures

## Introduction
This repo consisting of 2 main parts:
1. Jupyter Notebook that includes training, testing and inference.
2. Command Line Application that could be used in training and prediction.

In this project, you'll train an **image classifier** to predict 102-class of flower species, and export the trained model, then using them for inference afterwards.

Once completing this project, you'll have a **software** that can be trained on any set of labelled images to predict any kind of images not just flower species which will be a powerful tool when integrating it with any kind of applications that require image prediction. 

By the end of this project you'll deal with **a user-friendly command line application** that anyone could use without any previous requirements.

## Prerequisites

1. If you use Anaconda, you could create an environment with all required packages directly from `req.txt` by using the command:
    ```
    $ conda create --name <env> --file req.txt
    ```
2. If you don't, here are the required packages
    * `cudatoolkit==8.0`
    * `numpy==1.13.3`
    * `pandas==0.22.0`
    * `python==3.6.9`
    * `pytorch==0.4.0`
    * `torchvision==0.2.1`
    
     First make sure you've tha lastest pip version by the command:
     ```
     python -m pip install --upgrade pip
     ```
     Then you could use pip to install the packages
     ```
     pip install python==3.6.9 numpy==1.13.3
     ```
    
    **Hint**: to install cunda use the command: 
    ```
    conda install cudatoolkit=8.0 -c pytorch
    ```
    or
    ```
    pip install cudatoolkit==8.0 -c pytorch
    ```

## 1. Jupyter Notebook
In order to avoid rendering problems you could check it out in [nbviewer](https://nbviewer.jupyter.org/github/AhMeDxHaMiDo/Image-Classifier/blob/AhMeDxHaMiDo-patch-1/Image-Classifier-Project.ipynb).

# That would change

## 2. Command Line Application
* Training
* Prdiction

### Training
You would use **`train.py`** file to train a new Deep Neural Network on a dataset of images and saves the model to a checkpoint.

Required Arguments
* `data_dir` >>> directory path of the datsets

Optional Arguments
* `s` or `save_dir`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--->&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;directory path to save the Trained Model inside it
* `a` or `arch`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--->&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;choosing a CNN Model Architecture -- Default = vgg19
* `l` or `learning_rate`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--->&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;choosing a learning rate for DNN -- Default = 0.0001
* `hidden_units`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--->&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;setting number of the hidden units of the hidden layers >>> Must be integers -- Default = 1024'
* `d` or `drop_prob`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--->&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;setting number of drop probabilities of the hidden units of the hidden layers -- Default = 0.2
* `e` or `epochs`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--->&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;choosing number of model trainings (epochs) >>> Must be integer -- Default = 20
* `g` or `gpu`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--->&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;choosing GPU for training or inference

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This will appear with six space characters in front of it

Basic Usage: 
```







The predict.py script successfully reads in an image and a
checkpoint then prints the most likely image class and it's
associated probability


## Instructions
1. All files must be at the same directory.
2. In case of Training: 
                        
        - the data set must be labeled and divided in folders where each folder is named by its class number.
        - Example:  flowers\train\52\image_04221.jpg
                    Image Name: image_04221.jpg             Class Number: 52            Dataset of: Training
2.

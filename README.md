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
    * cudatoolkit==8.0
    * numpy==1.13.3
    * pandas==0.22.0
    * python==3.6.9
    * pytorch==0.4.0
    * torchvision==0.2.1
    
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

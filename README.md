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

### 1. Environment
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
    * `Image`
    
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
### 2. Data Pre-Processing
You could work on the flowers dataset or any other one but note that the dataset must be labeled and divided in folders where each folder is named by its class number.

**Example**

* flowers\train\52\image_04221.jpg

    + Image Name: image_04221.jpg
    + Class Number: 52
    + Dataset of: Training
    
* flowers\valid\1\image_06756.jpg

    + Image Name: image_06756.jpg
    + Class Number: 1
    + Dataset of: Validation

### 3. Label Mapping
You'll also need to load in a mapping from category label to category name. You can find this in the file cat_to_name.json in case of usinf flowers dataset. It's a JSON object which you can read in with the json module. This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

## 1. Jupyter Notebook
In order to avoid rendering problems you could check it out in [nbviewer](https://nbviewer.jupyter.org/github/AhMeDxHaMiDo/Image-Classifier/blob/AhMeDxHaMiDo-patch-1/Image-Classifier-Project.ipynb).

# That would change

## 2. Command Line Application
* Training
* Prdiction

### Training
You would use **`train.py`** file to train a new Deep Neural Network on a dataset of images and saves the model to a checkpoint.

* **Required Arguments**
    * `data_dir` ---> directory path of the datsets.

* **Optional Arguments**
    * `-s` or `--save_dir` ---> directory path to save the Trained Model inside it. -- Default = work directory
    * `-a` or `--arch` ---> choosing a CNN Model Architecture. -- Default = vgg19
    * `-l` or `--learning_rate` ---> choosing a learning rate for DNN. -- Default = 0.0001
    * `--hidden_units` ---> number of hidden units of hidden layers (Must be integers). -- Default = 1024
    * `-d` or `--drop_prob` ---> drop probability of the hidden units of the hidden layers. -- Default = 0.2
    * `-e` or `--epochs` ---> choosing number of model trainings (Must be integer). -- Default = 20
    * `-g` or `--gpu` ---> choosing GPU for training or inference.

* **Basic Usage**
```
    python train.py datasets_directory
```
* **Other Examples**
```
    python train.py datasets_directory -s checkpoints_directory --arch vgg16
```
```
    python train.py datasets_directory -l 0.001 --hidden_units 2048 512
```
```
    python train.py datasets_directory -d 0.1 -e 10 -g
```

* **Supported CNN Architectures**

    | Architectures |
    | ---- |
    | [AlexNet](https://arxiv.org/abs/1404.5997) |
    | [VGG11](https://arxiv.org/pdf/1409.1556.pdf) |
    | [VGG13](https://arxiv.org/pdf/1409.1556.pdf) |
    | [VGG16](https://arxiv.org/pdf/1409.1556.pdf) |
    | [VGG19](https://arxiv.org/pdf/1409.1556.pdf) |
    | [ResNet18](https://arxiv.org/pdf/1512.03385.pdf) |
    | [ResNet34](https://arxiv.org/pdf/1512.03385.pdf) |
    | [ResNet50](https://arxiv.org/pdf/1512.03385.pdf) |
    | [ResNet101](https://arxiv.org/pdf/1512.03385.pdf) |
    | [ResNet152](https://arxiv.org/pdf/1512.03385.pdf) |
    | [DenseNet121](https://arxiv.org/pdf/1608.06993.pdf) |
    | [DenseNet161](https://arxiv.org/pdf/1608.06993.pdf) |
    | [DenseNet169](https://arxiv.org/pdf/1608.06993.pdf) |
    | [DenseNet201](https://arxiv.org/pdf/1608.06993.pdf) |
* **Output**
    
    * While Training: Printing out current epoch, training loss, validation loss, and validation accuracy.
        * Ex: Epoch: 8/8..  Training Loss: 0.599..  Validation Loss: 0.782..  Validation Accuracy: 0.809
    * After Training: A checkpoint that contains the trained DNN wights, biases, and hyper parameters.
        * Ex :resnet18.pth

### Prediction
You would use **`predict.py`** file to predict the class of an image using the checkpoint of any saved model, and the probability of the top most likely classes.
    

- Predict flower name from an image with **predict.py** along with the probability of that name. That is you'll pass in a single image /path/to/image and return the flower name and class probability
  - Basic usage: ```python predict.py /path/to/image checkpoint```
  - Options:
    - Return top K most likely classes: ```python predict.py input checkpoint ---top_k 3```
    - Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_To_name.json```
    - Use GPU for inference: ```python predict.py input checkpoint --gpu```


* **Required Arguments**
    * `input` ---> path of the flower that you want to predict its label.
    * `checkpoint` ---> path of the trained DNN Model.

* **Optional Arguments**
    * `-k` or `--top_k` ---> choosing top K most likely classes. -- Default = 1
    * `-c` or `--category_names` ---> choosing a mapping of categories to real names
    * `-g` or `--gpu` ---> choosing GPU for training or inference.

* **Basic Usage**
```
    python predict.py input_image_path checkpoint
```
* **Other Examples**
```
    python predict.py input_image_path checkpoint -g --top_k 3
```
```
    python predict.py input_image_path checkpoint --category_names cat_To_name.json
```
```
    python predict.py input_image_path checkpoint -g -c cat_To_name.json -k 3
```

* **Output**
Printing the most likely image class and it's associated probability.


## License
[MIT](https://choosealicense.com/licenses/mit/)

*Inspired by Udacity AI Programming with Python Nanodgree*

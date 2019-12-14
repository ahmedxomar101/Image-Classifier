import sys
import json
from PIL import Image

import numpy as np

import torch
from torchvision import transforms, models

from train_functions import Network


def load_checkpoint(filepath, device):
    ''' Builds a function to load the model, returns the model.
        
        Arguments
        ---------
        filepath: string, the path of the model checkpoint.
        device: string, the device which the user want to use.
    '''

    print("Loading the Model ..")
    
    # Using try statement here is to handle the error, if the filepath of the model is not found, then exit.
    try:
        if device == 'cuda':
            # Using try statement here is to check if the input device is GPU, 
            # and handle the error if it's not available
            try:
                checkpoint = torch.load(filepath)
            # If there's a problem with the GPU or it's not available right now.
            # That will print the error and its number autotomatically.
            except RuntimeError:
                # Exit the script!
                sys.exit(0)
        # If the user input was to use the CPU, then we need to force the all tensors to be on CPU,
        # if they have been trained on GPU by the following command.
        else:
            checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    # Handling the error, if the filepath of the model is not found    
    except FileNotFoundError:
        print("The input path of the Model is Undefined!")
        print("Please try again with the correct path of the model checkpoint!")
        # Exit the script.
        sys.exit(0)
        
    # using the pre-trained Network
    model = checkpoint['model']
    
    # Freezing the parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    # Creating the Feedforward Classifier
    classifier = Network(input_size = checkpoint['input_size'],
                         output_size = checkpoint['output_size'],
                         hidden_layers = checkpoint['hidden_layers'], 
                         drop_p = checkpoint['drop_p'])

    # Replacing pre-trained calssifier by ours.
    model.classifier = classifier
    # Loading the weights and biases.
    model.load_state_dict(checkpoint['state_dict'])
    # Loading the classes to indices of our saved model.
    model.class_to_idx = checkpoint['class_to_idx']
    print("The Model is successfully Loaded.")
    print()
    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image and converts  into an object 
        that can be used as input to a trained model, returns an Numpy array
    '''
    
    im = Image.open(image_path)
    # Process a PIL image for use in a PyTorch model
    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.RandomResizedCrop(244),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])
    preprocessed_im = preprocess(im)
    
    return preprocessed_im

def predict(image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model,
        returns the probabilities and classes of most likely (k) predicted classes.
        
        Arguments
        ---------
        image_path: string, path of the image.
        model: the pre-trained model.
        device: string, the used device for inference which the user passed.
        topk: integer, number of most likely (k) classes.
    '''
    
    # Move model to the device
    model.to(device)
    # Model in inference mode, dropout is off
    model.eval()
    
    image = process_image(image_path)
    #print(image.size()) >>> torch.Size([3, 244, 244])
    image.unsqueeze_(0) 
    #print(image.size()) >>> torch.Size([1, 3, 244, 244])
    
    # Move image tensors to the device.
    image = image.to(device)
    
    # Turn off gradients for testing saves memory and computations, so will speed up inference.
    with torch.no_grad():
        # Forward pass through the network to get the outputs.
        prediction = model.forward(image)
    # Take exponential to get the probabilities from log softmax output.
    ps = torch.exp(prediction)
    # The most likely (topk) predicted prbabilities with their indices.
    probs, top_k_indices = ps.topk(topk)
    
    # Extracting the classes from the indices.
    classes = []
    for indice in top_k_indices.cpu()[0]:
        classes.append(list(model.class_to_idx)[indice.numpy()]) # Take the class from the index
    
    return probs.cpu()[0].numpy(), classes


# TODO: Display an image along with the top 5 classes


def predict_classes_names(cat_to_name, classes_output):
    ''' Mapping the category labels to category names, returns the category names.
        
        Arguments
        ---------
        cat_to_name: dict, label mapping from category label to category name. 
        classes_output: list, The most likely (k) predicted classes labels.
    '''

    # Opening the json file by using with statemnt to be closed after finishing.
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)

    # Creating the a list contains the category names from the classes labels.
    classes_names = []
    for i in classes_output:
        classes_names.append(cat_to_name[str(i)])
    
    return classes_names

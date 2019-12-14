import os
import sys
from pathlib import Path

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models

from utility_functions import choosing_arch


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output log softmax.
            Arguments
            ---------
            self: all layers
            x: tensor vector
        '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout.
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
    

def validation(model, validloader, criterion, device):
    ''' Builds a feedforward network with arbitrary hidden layers, 
        returns the validation loss and  validation accuracy.
        
        Arguments
        ---------
        model: the pre-trained model.
        validloader: generator, the validation dataset.
        criterion: loss function.
        device: the used device for the training [GPU or CPU].
    '''

    # Initiate the validation accuracy & validation loss with 0.
    valid_accuracy = 0
    valid_loss = 0
    # Move model to the device
    model.to(device)
    
    # Looping through the data batches.
    for inputs, labels in validloader:
        # Move input and label tensors to the device.
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass through the network.
        output = model.forward(inputs)
        # Increase the validation loss by the loss of the predicted output with the labels.
        valid_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, so take exponential to get the probabilities.
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label.
        equality = (labels.data == ps.max(dim=1)[1])
        # Accuracy is number of correct predictions divided by all predictions, so we just take the mean.
        valid_accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, valid_accuracy


def training(model, criterion, optimizer, device, trainloader, validloader, epochs, print_every=40):
    ''' Builds a feedforward network with arbitrary hidden layers.
        
        Arguments
        ---------
        model: the pre-trained model.
        optimizer: which we will take a step with it to update the weights.
        criterion: loss function.
        device: the used device for the training [GPU or CPU].
        trainloader: generator, the training dataset.
        validloader: generator, the validation dataset.
        epochs: integer, number of trainings.
        print_every: integer, printing the updates on loss & accuracy every print_every value.
    '''

    steps = 0
    running_loss = 0
    
    # Using try statement to deal with the device errors.
    try:
        # Move model to the device
        model.to(device)
    # If there's a problem with the driver.
    except AssertionError:
        print("Error Loading NVIDIA driver on your system. \nPlease check that you have an NVIDIA GPU and installed a driver from\nhttp://www.nvidia.com/Download/index.aspx")
        # Exit the script.
        sys.exit(0)

    # If there's a problem with the GPU or it's not available right now.
    # That will print the error and its number autotomatically.
    except RuntimeError:
        sys.exit(0)

    print("Model Training Started ...")
    print()
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for inputs, labels in trainloader:
            
            # Move input and label tensors to the device.
            inputs, labels = inputs.to(device), labels.to(device)
            
            steps += 1
            
            # zero-ing the accumalated gradients.
            optimizer.zero_grad()
            
            # Forward pass through the network
            output = model.forward(inputs)
            # Calculate the loss
            loss = criterion(output, labels)
            # Backward pass through the network 
            loss.backward()
            # Take a step with the optimizer to update the weights
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients for validation saves memory and computations, so will speed up inference
                with torch.no_grad():
                    valid_loss, valid_accuracy = validation(model, validloader, criterion, device)
                
                # Displaying the validation loss and accuracy during the training. 
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(valid_accuracy/len(validloader)))
                
                running_loss = 0
                
                # Make sure dropout and grads are on for training
                model.train()
    print("Model Training Finished")
    print()


def checkpoints_save(model, image_datasets, hyper_parameters, arch, save_dir):
    ''' Saving the model, weights, biases, mapping of classes to indices, 
        and hyper parameters to rebuild the model.
    '''
    print("Model Saving Started ..")
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'input_size': hyper_parameters['input_size'],
                  'output_size': hyper_parameters['output_size'],
                  'hidden_layers': hyper_parameters['hidden_layers'],
                  'drop_p': hyper_parameters['drop_p'],
                  'class_to_idx': model.class_to_idx,
                  'model': choosing_arch(arch),
                  'state_dict': model.state_dict()}


    # Check if the save_dir ends with '/' or not to concatenate the checkpoint name with it.
    if not save_dir.endswith('/'):
        save_dir += '/'

    filepath = save_dir + arch + '.pth'

    # Loop from 0 to 99 to check if there's another file in the same of ours or not, 
    # if there is, add a number to the name and repeat till reaching a name 
    # that is not existed, then use it.
    for i in range(100):
        my_file = Path(filepath)
        if my_file.is_file():
            filepath = save_dir + arch + "_" + str(i+1) + '.pth'
        else:
            break
    # Using try statement to handle if the input Save Directory doesn't exist by creating it.
    try:
        torch.save(checkpoint, filepath)
    # If the input Save Directory is not found, create it.
    except FileNotFoundError:
        print()
        print("Save Directory of the trained model is not Found!")
        print("Creating the Directory ..")
        print()
        os.mkdir(save_dir)
        torch.save(checkpoint, filepath)

    
    print("The Trained Model is successfully saved.")

def testing(model, testloader, device):
    ''' Builds a function to test the model, returns the testing accuracy.
        
        Arguments
        ---------
        model: the pre-trained model.
        testloader: generator, the testing dataset.
    '''

    # Model in inference mode, dropout is off
    model.eval()
    # Move model to the device
    model.to(device)
    accuracy = 0
    
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Turn off gradients for testing saves memory and computations, so will speed up inference
        with torch.no_grad():
            # Forward pass through the network to get the outputs
            output = model.forward(inputs)
        
        # take exponential to get the probabilities from log softmax output.
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    test_accuracy = accuracy / len(testloader)
    
    return test_accuracy


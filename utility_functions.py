import argparse
import torch
from torchvision import datasets, transforms, models


def get_input_args_train():
    ''' Getting arguments from the user for the train.py script.
    '''
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Train a Deep Neural Network for Flowers Classification Using CNN Model Architecture')
    
    # Create command line arguments using add_argument() from ArguementParser method

    # required arguments
    parser.add_argument('data_dir', type = str, help = 'directory path of the datsets')


    # optional arguments
    parser.add_argument('-s','--save_dir', type = str, default = './', 
                        help = 'directory path to save the Trained Model inside it')
    
    parser.add_argument('-a','--arch', type = str, default = 'vgg19', 
                        help = 'choosing a CNN Model Architecture -- Default = vgg19')
    
    parser.add_argument('-l','--learning_rate', type = float, default = 0.0001, 
                        help = 'choosing a learning rate for DNN -- Default = 0.0001')

    parser.add_argument('--hidden_units', nargs='*', type = int, default = [1024],
                        help = 'setting number of the hidden units of the hidden layers >>> Must be integers -- Default = 1024')

    parser.add_argument('-d','--drop_prob', type = float, default = 0.2,
                        help = 'setting number of drop probabilities of the hidden units of the hidden layers -- Default = 0.2')

    parser.add_argument('-e','--epochs', type = int, default = 20, 
                        help = 'choosing number of model trainings (epochs) >>> Must be integer -- Default = 20')
    
    parser.add_argument('-g','--gpu', action='store_true', help = 'choosing GPU for training or inference')

    return parser.parse_args()


def get_input_args_predict():
    ''' Getting arguments from the userfor the predict.py script.
    '''    
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Train a Deep Neural Network for Flowers Classification Using CNN Model Architecture')
    
    # Create command line arguments using add_argument() from ArguementParser method

    # required arguments
    parser.add_argument('input', type = str, help = 'path of the flower that you want to predict its label')
    parser.add_argument('checkpoint', type = str, help = 'path of the trained DNN Model')
    

    # optional arguments
    parser.add_argument('-k','--top_k', type = int, default = 1, 
                        help = 'choosing top K most likely classes')

    parser.add_argument('-c','--category_names', type = str, 
                        help = 'choosing a mapping of categories to real names')

    parser.add_argument('-g','--gpu', action='store_true', help = 'choosing GPU for training or inference')

    return parser.parse_args()



def preprocessing_datasets(data_dir):
    ''' Preprocessing for all datasets by applying multiple transforms, 
        returns the dataloaders and the image_datasets.
        
        Arguments
        ---------
        data_dir: string, the directory of the datasets.
    '''

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Defining transforms for the training, validation, and testing sets
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                               transforms.RandomResizedCrop(244),
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225])
                                                              ]),
                       'valid': transforms.Compose([transforms.Resize(256),
                                                               transforms.RandomResizedCrop(244),
                                                               transforms.ToTensor(),
                                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225])
                                                              ])}

    # Loading the datasets with ImageFolder.
    image_datasets = {'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                      'test': datasets.ImageFolder(test_dir, transform=data_transforms['valid'])
                     }

    # Using the image datasets and the transforms to define the dataloaders.
    dataloaders = {'trainloader': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
                   'validloader': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
                   'testloader': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
                  }
    
    return dataloaders, image_datasets


def choosing_arch(arch):
  ''' Choosing the desired architecture of the pre-trained model, returns the pre-treained model.
        
      Arguments
      ---------
      arch: string, name of the pre-trained architecture.
  '''

  if arch == 'alexnet' or arch == 'AlexNet':
      model = models.alexnet(pretrained=True)

  elif arch == 'vgg11':
      model = models.vgg11(pretrained=True)
  elif arch == 'vgg13':
      model = models.vgg13(pretrained=True)
  elif arch == 'vgg16':
      model = models.vgg16(pretrained=True)
  elif arch == 'vgg19':
      model = models.vgg19(pretrained=True)

  elif arch == 'resnet18':
      model = models.resnet18(pretrained=True)
  elif arch == 'resnet34':
      model = models.resnet34(pretrained=True)
  elif arch == 'resnet50':
      model = models.resnet50(pretrained=True)
  elif arch == 'resnet101':
      model = models.resnet101(pretrained=True)
  elif arch == 'resnet152':
      model = models.resnet152(pretrained=True)

  elif arch == 'densenet121':
      model = models.densenet121(pretrained=True)
  elif arch == 'densenet161':
      model = models.densenet161(pretrained=True)
  elif arch == 'densenet169':
      model = models.densenet169(pretrained=True)
  elif arch == 'densenet201':
      model = models.densenet201(pretrained=True)
  elif arch == 'inception_v3':
      model = models.inception_v3(pretrained=True)
  else:
      recall = input("Undefined!\nPlease Enter Another Architecture:\n")
      choosing_arch(recall)
  return model




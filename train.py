from torch import nn, optim
from collections import OrderedDict

from utility_functions import get_input_args_train, preprocessing_datasets, choosing_arch
from train_functions import Network, training, checkpoints_save, testing
from workspace_utils import active_session
                

def main():
    # Calling the get_input_args_train function to get the input arguments.
    in_arg = get_input_args_train()

    # Obtaining the dataloaders and image_datasets by passing the datasets directory to 
    # preprocessing_datasets function.
    data_dir = in_arg.data_dir
    dataloaders, image_datasets = preprocessing_datasets(data_dir)

    # Obtaining the pre-trained model by passing the architecture to the choosing_arch function.
    arch = in_arg.arch
    print('Model Architecture is Loading ..')
    model = choosing_arch(arch)


    # Using try statement to handel the error of the number of hidden units of the classifier
    # Because if the classifier has a single layer, it will contain the number of them directly,
    # but if it has multiple layers, they will contain them from the first index.
    try:
        input_size = model.classifier.in_features
    # If the classifier has multiple layers, take the in_features from the first one.
    except AttributeError:
        input_size = model.classifier[0].in_features

    # Setting all hyper parameters in a dictionary to ease the dealing.
    # All parameters are arguments passed from the user.
    # in_arg.classes_n is default as 102 for the flowers problem or could
    # be any other number of classes for any type of other problems.
    hyper_parameters = {'input_size': input_size,
                        'output_size': in_arg.classes_n,
                        'hidden_layers': in_arg.hidden_units,
                        'drop_p': in_arg.drop_prob,
                        'learn_rate': in_arg.learning_rate,
                        'epochs': in_arg.epochs
                        }

    # Freezing the parameters so we don't backprop through them, 
    # we will backprop through the classifier parameters only later
    for param in model.parameters():
        param.requires_grad = False

    # Checking if the input hidden_layers by the user is not empty!
    if len(hyper_parameters['hidden_layers']) != 0:
        # Creating Feedforward Classifier
        classifier = Network(input_size = hyper_parameters['input_size'], 
                             output_size = hyper_parameters['output_size'], 
                             hidden_layers = hyper_parameters['hidden_layers'], 
                             drop_p = hyper_parameters['drop_p'])

    # If the user pass nothing for the hidden units, it means we will create the clssifier 
    # without hidden layers, so it will only consist from a single layer of the linear
    # transformations of the in_features of the model with the 102 the output classes,
    # then apply the log softmax activation function on them.
    else:
        ## Build a feed-forward network to create the classifier.
        classifier = nn.Sequential(OrderedDict([
                          ('output', nn.Linear(input_size, in_arg.classes_n)),
                          ('log_softmax', nn.LogSoftmax(dim=1))]))


    # Replacing the pre-trained calssifier of the model by the classifier we've just created.
    model.classifier = classifier

    # Define the criterion (Loss function). 
    criterion = nn.NLLLoss()
    # Define the optimizer. Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=hyper_parameters['learn_rate'])

    # checking what device the user want to train by.
    if in_arg.gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    with active_session():
        # do long-running work here

        # Train the model with a pre-trained network.
        training(model, criterion, optimizer, device,
                 trainloader = dataloaders['trainloader'],
                 validloader = dataloaders['validloader'],
                 epochs = hyper_parameters['epochs'])


    # Obtaining the save directory from the user then,
    # saving the model, weights, biases, mapping of classes to indices, 
    # and hyper parameters to rebuild the model.
    save_dir = in_arg.save_dir
    checkpoints_save(model, image_datasets, hyper_parameters, arch, save_dir)


    # Calculate the avergae accuracy of our pre-trained model.
    print()
    print("Model Accuracy Testing Started ...")
    test_sum = 0
    for i in range(10):
        test_accuracy = testing(model, dataloaders['testloader'], device)
        test_sum += test_accuracy
        print("Test Accuracy({}): {:.3f}".format(i+1, test_accuracy))
    
    avg_test_accuracy = test_sum/10
    print()
    print("Average Test Accuracy: {:.3f}".format(avg_test_accuracy))
    

if __name__ == "__main__":
    main()

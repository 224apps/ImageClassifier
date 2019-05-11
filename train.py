
###################################
# Author: Abdoulaye Diallo
# File_name: train.py
# Title: Image Training classifier.
###################################
#Imported libraries
import argparse
from os.path import isdir
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict


# Function arg_parser() parses keyword arguments from the command line
def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, help= 'what kind of pretrained architecture to use as str ')
    parser.add_argument('--learning_rate', type=float, help='Define gradient descent learning rate as float')
    parser.add_argument('--save_dir', type=str, help='save trained checkpoint to this directory as str ')
    parser.add_argument('--hidden_units', type=int, help='Hidden units for DNN classifier as int')
    parser.add_argument('--epochs', type=int, help='# of epochs for training as int')
    parser.add_argument('--gpu', action="store_true",  help='Use GPU + Cuda  to traimn the model')
    args = parser.parse_args()
    return args


# DONE: Define your transforms for the training, validation, and testing sets and define custom functions.
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
}
# DONE: Load the datasets with ImageFolder
image_datasets = {
    'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}

# DONE: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=50, shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=50),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=50)
}

#  these Functions perform training transformations on the dataset
def train_trans(train_dir):
    # Define the transform and then load the data.
    train_transforms =  data_transforms ['train']
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

def test_trans(test_dir):
     # Define the transform and then load the data.
    test_transforms = data_transforms ['test']
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader


#  Check GPU and make decision using the Device selecgted or use the CPU.
def check_gpu(gpu):
    if not gpu:
        return torch.device("cpu")
    #chose cuda: 0 if there is a GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Print result
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device

# primaryloader_model(architecture) downloads model (primary) from torchvision
def primaryloader_model(architecture):
    #if there is not an architecture chosen, load the default.
    if architecture == 'resnet':
        model = models.resnet18(pretrained=True)
        num_in_features = model.fc.in_features
    elif architecture == 'vgg':
        model = models.vgg16(pretrained=True)
        num_in_features = model.classifier[0].in_features
    else: 
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    return model

# Function initial_classifier(model, hidden_units) creates a classifier with the corect number of input layers
def initial_classifier(model, hidden_units):
    # Check that hidden layers has been input
    if type(hidden_units) == type(None): 
        #hyperparamters
        hidden_units = 4096 
        print('Hidden Units specificed: 4096.')
    #Input Layers
    input_features = model.classifier[0].in_features
    
    # Define Classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

# validating the  training against testloader to return loss and accuracy
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

#  representing the training of the network mode
def network_trainer(Model, Trainloader, Testloader, Device,  Criterion, Optimizer, Epochs, Print_every, Steps):
    # Check Model Kwarg
    if type(Epochs) == type(None):
        Epochs = 5
        print("Number of Epochs specificed as 5.")    
 
    print("Training process initializing .....\n")

    # Train Model
    for e in range(Epochs):
        running_loss = 0
        Model.train() # Technically not necessary, setting this for good measure
        
        for ii, (inputs, labels) in enumerate(Trainloader):
            Steps += 1
            inputs, labels = inputs.to(Device), labels.to(Device)
            Optimizer.zero_grad()
            # Forward and backward passes
            outputs = Model.forward(inputs)
            loss = Criterion(outputs, labels)
            loss.backward()
            Optimizer.step()
        
            running_loss += loss.item()
        
            if Steps % Print_every == 0:
                Model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(Model, dataloaders['valid'], Criterion)
                print("Epoch: {}/{} ------ ".format(e+1, epochs),
                  "Training Loss: {:.4f} ------ ".format(running_loss/print_every),
                  "Validation Loss: {:.4f} ------ ".format(valid_loss/len(dataloaders['test'])),
                  "Validation Accuracy: {:.4f}".format(accuracy/len(dataloaders['test'])))
                running_loss = 0
                Model.train()

    return Model

#Function to validate the above model on test data images
def validate_model(Model, Testloader, Device):
   # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))

# Function initial_checkpoint(Model, Save_Dir, Train_data) saves the model at a defined checkpoint
def initial_checkpoint(Model, Save_Dir, Train_data):
       
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(Save_Dir):
            # Create `class_to_idx` attribute in model
            Model.class_to_idx = Train_data.class_to_idx
            
            # Create checkpoint dictionary
            checkpoint = {'architecture': Model.name,
                          'classifier': Model.classifier,
                          'class_to_idx': Model.class_to_idx,
                          'state_dict': Model.state_dict()
                         }
            
            # Save checkpoint
            torch.save(checkpoint, 'my_checkpoint.pth')

        else: 
            print("Directory not found, model will not be saved.")


# =============================================================================
# Main Function
# =============================================================================

def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = test_trans(train_dir)
    valid_data = train_trans(valid_dir)
    test_data = train_trans(test_dir)
    
    train_loader = data_loader(train_data)
    valid_loader = data_loader(valid_data, train=False)
    test_loader = data_loader(test_data, train=False)
    
    # Load Model
    model = primaryloader_model(architecture=args.arch)
    
    # Build Classifier
    model.classifier = initial_classifier(model,  hidden_units=args.hidden_units)
     
    # Check for GPU
    device = check_gpu(gpu=args.gpu);
    
    # Send model to device
    model.to(device);
    
    # Check for learnrate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Define deep learning method
    print_every = 30
    steps = 0
    
    # Train the classifier layers using backpropogation
    trained_model = network_trainer(model, train_loader, valid_loader,  device, criterion, optimizer, args.epochs, print_every, steps)
    
    print("=========================Training process is completed. ======================================\n")
    # Validate and save the model
    validate_model(trained_model, testloader, device)
    initial_checkpoint(trained_model, args.save_dir, train_data)
    
# Run Program
if __name__ == '__main__':
    main()
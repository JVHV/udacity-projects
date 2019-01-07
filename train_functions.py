import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sb
from collections import OrderedDict

from PIL import Image

#def test_parse(num=5):
#    print ('This prints your number {}. If you see this the test worked'.format(num))

def network_transforms(data_dir = 'flowers'):
    #data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    #data_transforms =                                   
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]) 
                                      
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=training_transforms)

    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms) 

    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    # TODO: Using the image datasets and the tranforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True) 
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size = 64)                                     
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)  

    return trainloader, validationloader, testloader, train_data
"""                     
def model_arch(arch = 'vgg16'):      
    if arch == 'vgg16':           
        model = models.vgg16(pretrained=True)
        print ("vgg16 selected succesfully") 
        return model
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        print ("densenet121 selected succesfully") 
        return model
    else:
        print("Please select arch = vgg16 or  arch = densenet121 for the architecture of your model") 
"""                
  
def model_classifier(hidden_units=1, arch ='vgg16', lr = 0.0001):
    if arch == 'vgg16':           
        model = models.vgg16(pretrained=True)
        print ("vgg16 selected succesfully") 
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        print ("densenet121 selected succesfully") 
    else:
        print("Please select arch = vgg16 or  arch = densenet121 for the architecture of your model") 
        
    for param in model.parameters():
        param.requires_grad = False 
    total_n_of_layers= hidden_units + 1
    drop = ('dropout', nn.Dropout(p=0.2))
    relu = ('relu', nn.ReLU())
    fc_number = 1
    layers = [] 
    if arch == 'vgg16':
        layer_input = 25088
        control_value = 4000
        
    elif arch == 'densenet121':
        layer_input = 1024
        
    stat = int((control_value-1000)/total_n_of_layers)
    var = control_value 
        
    while total_n_of_layers != 0 : 
        layers.append(('fc{}'.format(fc_number), nn.Linear(layer_input, var)))
        layers.append((drop))
        layers.append((relu)) 
        layer_input = var
        var -= stat
        fc_number += 1
        total_n_of_layers -= 1
                 
    layers.append(('fc{}'.format(fc_number), nn.Linear(layer_input, 102)))
    layers.append(('output', nn.LogSoftmax(dim=1)))
    model.classifier = nn.Sequential(OrderedDict(layers))
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    return model, optimizer, layers

def validation(model,dataloader, criterion):
    test_loss = 0
    accuracy = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        if device == torch.device("cuda:0"):
             accuracy+= equality.type(torch.cuda.FloatTensor).mean()
        else:
             accuracy+= equality.type(torch.FloatTensor).mean() 
        
    return test_loss, accuracy
        
def model_training(model, trainloader, validationloader, device, epochs, optimizer):
    if device =='gpu':
        device = torch.device('cuda')
        print (' cuda gpu is activated')
    steps = 0
    model.to(device) 
    criterion = nn.NLLLoss()
    running_loss = 0
    print_every = 40
    start = time.time()
    print ('-----------Watch how I learn faster than you.---------------') 
    for e in range(epochs):
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
            if steps % print_every == 0:
                
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validationloader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(validationloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validationloader)))

                running_loss = 0

                # training back on
                model.train()
               
    final_time = time.time() - start
    print('Completed in {:.0f}m {:.7f}s'.format(final_time // 60, final_time % 60))

    
def save_checkpoint(model, save_directory, optimizer, train_data, epochs, arch, layers):
    checkpoint = {'layers': layers,
                 'arch': arch,
                 'optimizer':optimizer.state_dict(),
                 'epochs': epochs,
                 'model.class_to_idx': train_data.class_to_idx,
                 'state_dict':model.state_dict()}
              

    torch.save(checkpoint, save_directory)
        


        
        
        
        
        
        
        
        
        
        
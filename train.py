import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd
import time
from collections import OrderedDict
import argparse
import train_functions as tf 

parser = argparse.ArgumentParser() 
#parser.add_argument('num', help='whatever')
parser.add_argument('--data_dir', help= "The directory of the dataset", default='/home/workspace/aipnd-project/flowers', type = str)
parser.add_argument('--arch', help= "The architecture of the network. vgg16 or densenet121", default='vgg16', type=str)
parser.add_argument('--learning_rates', help= "The learning rate for the optimizer", default=0.0001, type= float) 
parser.add_argument('--hidden_units', help = "Number of hidden units", default=1, type = int)
parser.add_argument('--epochs', help = "Number of epochs", default=2, type = int)
parser.add_argument('--gpu', help = 'Activate the gpu, if you want gpu mode enabled type:"gpu", else: "cpu" ', default='gpu', type = str)               
parser.add_argument('--save_dir', help = "The path of the directory to save the trained network", default='checkpoint.pth', type = str)
                    
args = parser.parse_args()
                    
data_dir = args.data_dir
arch = args.arch
lr = args.learning_rates
hidden_units = args.hidden_units
epochs = args.epochs
device = args.gpu
save_directory = args.save_dir

                    
trainloader, validationloader, testloader, train_data = tf.network_transforms(data_dir) 
                    
model, optimizer, layers = tf.model_classifier(hidden_units, arch, lr)                    
                    
tf.model_training(model, trainloader, validationloader, device, epochs, optimizer)
                    
tf.save_checkpoint(model, save_directory, optimizer, train_data, epochs, arch, layers)
                    
print ("It's all set") 
                    
                    
                    
                    
                    
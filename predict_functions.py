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
import json

def to_open_json(class_to_cat):
    with open(class_to_cat, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
            param.requires_grad = False
    model.class_to_idx = checkpoint['model.class_to_idx']
    
    model.classifier = nn.Sequential(OrderedDict(checkpoint['layers']))
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    img_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
    
    final_image = img_transform(img) 
    
    #Looking at size of final_image it seems channel is in first dimension now.
    
    return final_image.numpy()  
    

def predict(model, image_path, topk, checkpoint):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    #process_image() gives a numpy array we need to convert it to a torch
    image = torch.from_numpy(process_image(image_path))
    image.unsqueeze_(0)
    model.eval()
    output = model.forward(image)
    ps = torch.exp(output)
    #Now I need to convert the indices to the class label using class_to_idx. The returned topk indices are the values
    # of the class_to_idx dictionary. Those keys are the keys to the flowers of the cat_to_name dictionary. We need to
    #get those keys so we can map it later to the names of the flowers.
    
    probabilities, indices = torch.topk(ps, topk, largest=True, sorted=True)
    
    probabilities = probabilities[0].detach().numpy().tolist()
    indices = indices[0].detach().numpy().tolist()
    # From https://stackoverflow.com/questions/483666/python-reverse-invert-a-mapping we get this way to reverse dict.
    #inv_topk = {v: k for k, v in model.class_to_idx.items()}
    inv_class_to_idx = {v: k for k, v in checkpoint['model.class_to_idx'].items()}
    classes = [inv_class_to_idx[i] for i in indices]
 
    return probabilities, classes
    
def display_flower_name(probabilities, classes, cat_to_name):
    flower_name = [cat_to_name[i] for i in classes]
    print ("The probabilities of your image to be one of these classes are:")
    for i in range (len(probabilities)):
        print ('{}  : {:.2f}%'.format(flower_name[i], probabilities[i] * 100))
    
    
    
    
    
    
    
    
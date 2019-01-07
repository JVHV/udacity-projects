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
import predict_functions as pf


parser = argparse.ArgumentParser() 
#parser.add_argument('num', help='whatever')
parser.add_argument('--image', help= "The image path", default='/home/workspace/aipnd-project/flowers/test/1/image_06743.jpg', type = str)
parser.add_argument('--checkpoint', help= "The checkpoint path", default='checkpoint.pth', type = str)
parser.add_argument('--gpu', help = 'Activate the gpu, if you want gpu mode enabled type:"gpu", else: "cpu" ', default='gpu', type = str)               
parser.add_argument('--topk', help = "The number of topk probabilities to be displayed", default= 1, type = int)
parser.add_argument('--class_to_cat', help = 'JSON file that maps the class values to other category names', default = 'cat_to_name.json', type = str)   

args = parser.parse_args()

image = args.image
checkpoint_path = args.checkpoint
device = args.gpu
topk = args.topk
class_to_cat = args.class_to_cat

#Loads and build the model from the path
cat_to_name = pf.to_open_json(class_to_cat) 

model, checkpoint = pf.load_checkpoint(checkpoint_path)

# Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
pf.process_image(image)

probabilities, classes = pf.predict(model, image, topk, checkpoint)

pf.display_flower_name(probabilities, classes, cat_to_name)
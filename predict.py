import matplotlib.pyplot as plt
import torch
import PIL
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import Image
import Helper
import numpy as np
import seaborn as sns
from collections import OrderedDict
import argparse


parser=argparse.ArgumentParser(description='Enter arguments for neural network:')
parser.add_argument('--image',
                    action='store',
                    default='flowers/test/20/image_04912.jpg',
                    help='Direct to image file')

parser.add_argument('--checkpoint',
                    action='store',
                    default='./project_checkpoint.pth',
                    help='Direct to checkpoint file')

parser.add_argument('--top_k',
                    action='store',
                    default=5,
                    type=int,
                    help='Choose top K classes')

parser.add_argument('--category_names',
                    default='cat_to_name.json',
                    help='Map of categories to names')

parser.add_argument('--gpu',
                    action='store_true',
                    default='gpu',
                    help='Use GPU when training')

in_args=parser.parse_args()
image =in_args.image
checkpoint =in_args.checkpoint
top_k =in_args.top_k  
category_names=in_args.category_names
json_file = in_args.category_names

with open(json_file, 'r') as f:
    cat_to_name = json.load(f)
                    
def load_checkpoint():
 
    checkpoint = torch.load(in_args.checkpoint)
    model=checkpoint['model']
    classifier = checkpoint['classifier']
    learning_rate= checkpoint['learning_rate']
    epochs = checkpoint['epoch']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):

    test_image = PIL.Image.open(image)

   
    orig_width, orig_height = test_image.size

    
    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]
        
    test_image.thumbnail(size=resize_size)

    
    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

   
    np_image = np.array(test_image)/255 

    
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
 
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

model = load_checkpoint()

def predict(image_path, model, top_k):
    
    model.to('cpu')
    
    # Set model to evaluate
    model.eval();

    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to('cpu')

  
    log_probability = model.forward(torch_image)


    linear_probability = torch.exp(log_probability).data

  
    top_probability, top_labels = linear_probability.topk(top_k)
    
    
    top_probability = np.array(top_probability.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probability, top_labels, top_flowers

im_path = in_args.image

probability, classes,t_flowers = predict(im_path, model,top_k)
print(probability)
print(classes)   
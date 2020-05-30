import matplotlib.pyplot as plt
import torch
import PIL
torch.cuda.is_available() 
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


def input_args():
    
    parser=argparse.ArgumentParser(description='Enter arguments for neural network:')
    parser.add_argument('image',action='store',dest='image',type='str',help='Direct to image file',required=True)
    parser.add_argument('checkpoint',action='store',dest='checkpoint',type='str',help='Direct to checkpoint file',required=True)
    parser.add_argument('--top_k',action='store',dest='top_k',type='int',default=5,help='Choose top K classes',required=True)
    parser.add_argument('--category_names',dest='category_names',default='cat_to_name.json',help='Map of categories to names')
    parser.add_argument('--gpu',action='store_true',dest='gpu',default='false',help='Use GPU when training (cuda)')
    print(parser.parse_args())
    return parser.parse_args()
                    
def load_checkpoint():
 
    checkpoint = torch.load("project_checkpoint.pth")
    
    model = models.densenet121(pretrained=True);
    
    for param in model.parameters(): param.requires_grad = False
    
    trained_model= torch.load("project_checkpoint.pth")
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    class_idx = trained_model['class_to_idx']
    
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

def predict(image_path, model, top_k=5):
    
    model.to('cpu')
    
    # Set model to evaluate
    model.eval();

   
    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path), 
                                                  axis=0)).type(torch.FloatTensor).to('cpu')

  
    log_probability = model.forward(torch_image)


    linear_probability = torch.exp(log_probability)

  
    top_probability, top_labels = linear_probability.topk(top_k)
    
    
    top_probability = np.array(top_probability.detach())[0] 
    top_labels = np.array(top_labels.detach())[0]
    
    
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probability, top_labels, top_flowers

probability, classes,t_flowers = predict(im_path, model)
print(probability)
print(classes)

im_path = 'flowers/test/20/image_04912.jpg'
model= models.densenet121(pretrained=True)

plt.figure(figsize = (6,10))
ax = plt.subplot(2,1,1)


flower_num = im_path.split('/')[2]
title_ = cat_to_name[flower_num]


img = process_image(im_path)
imshow(img, ax, title = title_);
class_idx=train_data.class_to_idx

probability, labels, flowers = predict(image_path, model, top_k=5 ) 

# Plot bar chart
plt.subplot(2,1,2)
sns.barplot(x=probability, y=flowers, color=sns.color_palette()[0]);
plt.show()

def main():
    
    args = input_args()
    
    model = load_checkpoint(args.checkpoint)
    
    im_path = process_image(args.image)
    
    probability, labels, flowers = predict(im_path, model, args.top_k) 
    
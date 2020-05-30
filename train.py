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

parser = argparse.ArgumentParser()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

parser.add_argument('data_dir',
                    action='store')

parser.add_argument('learning_rate',
                    default=0.001,
                    type=float,
                    help='Define learning rate gradient descent')

parser.add_argument('--epochs',
                    action='store',
                    type=int,
                    default=8,
                    help='Define epochs for training')

parser.add_argument('--gpu',
                    action='store',
                    dest='gpu',
                    help='Use GPU for training')

parser.add_argument('--save-dir',
                    action='store',
                    dest='save_dir',
                    type=str,
                    help='Set directory for the checkpoint.if not done all work will be lost')
                    
                    
train_transforms=transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
 
train_data=datasets.ImageFolder(data_dir +'/train',transform=train_transforms)
test_data=datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=96, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader= torch.utils.data.DataLoader(test_data, batch_size=64)

model = models.densenet121(pretrained=True)
model.name='densenet'

for param in model.parameters():
    param.requires_grad = False
    
import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
device = torch.device("cuda:0")

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024,1024)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(0.0)),
                          ('fc2', nn.Linear(1024,102)),
                          ('logsoftmax', nn.LogSoftmax(dim=1))]))

model.classifier = classifier


print(model)

learning_rate = 0.001
optimizer = optim.Adam(model.classifier.parameters(), 0.001)
device=torch.device('cuda:0')
model.to(device)
criterion = nn.NLLLoss()

# Implement a function for the validation pass
def validation(model, testloader, criterion):
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

print_every=32
steps =0
print("Initialize training .....\n")
epochs = 8

for e in range(epochs):
    running_loss = 0
    model.train() 
    
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
     
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()

            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion)
            
            print("Epoch: {}/{} | ".format(e+1, epochs),
                  "Training Loss: {:.4f} | ".format(running_loss/print_every),
                  "Validation Loss: {:.4f} | ".format(valid_loss/len(testloader)),
                  "Validation Accuracy: {:.4f}".format(accuracy/len(testloader)))
            
            running_loss = 0
            model.train()
            
 # TODO: Do validation on the test set
model.eval()
    
with torch.no_grad():
    _, accuracy = validation(model, testloader, criterion)
                
print("Test Accuracy on the model: {:.2f}%".format(accuracy*100/len(testloader)))

model.class_to_idx = train_data.class_to_idx

checkpoint = {'classifier': model.classifier,
              'structure':model.name,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict(),
             }

torch.save(checkpoint, 'project_checkpoint.pth')



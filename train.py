#!/usr/bin/env python3
# Imports here
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import os.path
from os import path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image

from collections import OrderedDict

import argparse
import json
from get_cmd_args import get_cmd_args
from print_functions import *
from time import time, sleep
from workspace_utils import keep_awake

default_data_dir = 'flowers'
num_of_classes = 102

 
def create_datasets():
    global image_dataset, train_dataset , valid_dataset , test_dataset
    all_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.Resize(255),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                     transforms.Normalize((0.485,0.456, 0.406),(0.229, 0.224, 0.225))])  


    test_transforms =transforms.Compose([transforms.CenterCrop(224),
                                     transforms.ToTensor(), 
                                     transforms.Normalize([0.485,0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


    valid_transforms =transforms.Compose([transforms.CenterCrop(224),
                                     transforms.ToTensor(), 
                                     transforms.Normalize([0.485,0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder
    ## datasets are global variable
    image_dataset = datasets.ImageFolder(data_dir, transform=all_data_transforms)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    return

def create_dataloaders():
    print("START create_dataloader();batch_size:{}".format(batch_size))
    global train_loader , valid_loader , test_loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print("END create_dataloader()")
    return

def create_classifier(vgg16_model , hidden_layers ):
    print("START create_classifier()")
    drop_out = 0.5
    in_features=vgg16_model.classifier[0].in_features
    print("IN_FEATURES:",in_features)
    hidden_layers = 512
    for param in vgg16_model.parameters():
        param.requires_grad = False

    if (hidden_layers == 512):
       intermediate_layers =256
    elif (hidden_layers == 2048):
        intermediate_layers =1024
    else:
        hidden_layers = 512
        intermediate_layers = 256
        

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(in_features, hidden_layers)),  ## 4096 is out features from VGG model
                          ('relu', nn.ReLU()),
                          ('dropout1' , nn.Dropout(drop_out)),
                          ('fc2', nn.Linear(hidden_layers, intermediate_layers)),  ## 4096 is out features from VGG model
                          ('relu2', nn.ReLU()),
                          ('droput2' , nn.Dropout(drop_out)),
                          ('fc3', nn.Linear(intermediate_layers, num_of_classes)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    print("END create_classifier()")
    return (classifier)

## validation function
def validation(model, loader, criterion, device):
    test_loss = 0
    accuracy = 0
    for images, labels in loader:

        ## images.resize_(images.shape[0], 784) ## not required as images are presized in loader
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint= torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.input_size = checkpoint['input_size']
    model.output_size = checkpoint['output_size']
    #   model.hidden_layers = checkpoint['hidden_layers']  ## ???
    return model
    

def main():
    global model , device , data_dir , train_dir , valid_dir , test_dir
    global batch_size
    # TODO 0: Measures total program runtime by collecting start time
    start_time = time()
    print("start time {}".format(start_time))
    
    in_arg = get_cmd_args()
    #check_command_line_arguments(in_arg)
    if (in_arg.data_dir != default_data_dir):
        print("ERROR ERROR only allowed data_dir is 'flowers'")
    
        
    data_dir = in_arg.data_dir;
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    gpu = in_arg.gpu
    arch = in_arg.arch
    save_dir = in_arg.save_dir ## to save checkpoint path
    lr = in_arg.learning_rate
    save_dir = in_arg.save_dir
    epochs = in_arg.epochs
    batch_size =in_arg.batch_size
    hidden_layers = in_arg.hidden_units
    ## create save_dir
    save_dir = os.path.join("/home/workspace/ImageClassifier/" , save_dir)
    #save_dir = "/home/workspace/ImageClassifier/" + save_dir
    if not os.path.isdir(save_dir ):
        os.mkdir(save_dir, mode = 0o755)
    ## create checkpoint file suffix
    time_suffix = datetime.now().timestamp
    print("Running train.py with:", 
              "\n    data_dir = ",in_arg.data_dir,
              "\n    gpu =" , in_arg.gpu ,
              "\n    arch =", in_arg.arch, "  learning_rate =", in_arg.learning_rate,
              "\n    save_dir=" , in_arg.save_dir,
              "\n    epochs =" , in_arg.epochs,
              "\n    batch_size =" , in_arg.batch_size)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    create_datasets()
    create_dataloaders()
        
    device = torch.device("cpu")
    cuda = torch.cuda.is_available()
    if (gpu and cuda):
        device = torch.device("cuda:0")
    print("CUDA:{}".format(cuda))
    
    if (arch == 'vgg13' or arch == 'vgg16'):
        model = models.vgg13(pretrained=True)
        no_input_layer = 25088
    elif (arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
        no_input_layer = 25088
       
    elif (arch == 'alexnet'):
       model = models.alexnet(pretrained=True)
       no_input_layer = 9216
    else:
       print("train.py does not support model:{}".format(arch))
       print("train.py supports only vgg13 , vgg16,alexnet")
       print ("Defaulting to vgg16")
       model = models.vgg16(pretrained=True)
       no_input_layer = 25088
    
    #model
    
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print("\nOur model:\n\n", model,'\n')
    print("State dict keys:\n\n" , model.state_dict().keys())

    print("DEVICE being used is:",device)
    
    model.classifier.out_features = num_of_classes
          
    model_classifier = create_classifier(model , hidden_layers )
            
    model.classifier = model_classifier
    print("DEVICE:{}".format(device))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    ## start Training
    print("Start training")
    ##  exit()
    steps = 0
    running_loss = 0
    print_every = 40
    for i in keep_awake(range(5)): 
        for e in range(epochs):
            model.train()
            for images, labels in train_loader:
                steps += 1
                images, labels = images.to(device), labels.to(device)
               
                optimizer.zero_grad()
        
                output = model.forward(images)
                output = output.to(device)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
        
                if steps % print_every == 0:
                    # Make sure network is in eval mode for inference
                    model.eval()
            
                     # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = validation(model, valid_loader , criterion , device)
                
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(valid_loader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(valid_loader)))
            
                    running_loss = 0
            
                    # Make sure training is back on
                    model.train()
    end_time = time()
    print("start time:{};end time:{}".format(start_time,end_time))
    print("End training")
    ## end training
    # TODO: Save the checkpoint 
    model.class_to_idx = train_dataset.class_to_idx
 
    checkpoint = {
        'arch': arch,
        'input_size':no_input_layer,
        'output_size':102,
        'class_to_idx': model.class_to_idx, 
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'classifier': model.classifier
    }
        #'hidden_layers':[each.out_features for each in vgg16_model.hidden_layers] gave following error
    ##AttributeError: 'VGG' object has no attribute 'hidden_layers'

    torch.save(checkpoint , '/home/workspace/ImageClassifier/' + save_dir +'/checkpoint.pth')
    print("END saving checkpoint")
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
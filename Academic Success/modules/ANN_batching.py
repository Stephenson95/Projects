# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:03:53 2024

@author: Stephenson
"""
#Import standard libraries
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

#Reproducibility
seed = 17
torch.manual_seed(seed)

#Set import path
import_path = r"C:\Users\{}\Documents\GitHub\Projects\Academic Success".format(os.getlogin())

#Import modules
sys.path.append(import_path + r'\modules')
from Dataloader import AcademicDataset
from Model import ANNBasic
from utils import train_batch, test, compute_results

#Load dataset
scaler = StandardScaler()
dataset = AcademicDataset(import_path, transform=scaler)

train_idx, test_idx = train_test_split(np.arange(len(dataset)),
                                             test_size=0.2,
                                             random_state=seed,
                                             shuffle=True,
                                             stratify=[label for _, label in dataset])

train_dataset = torch.utils.data.Subset(dataset, train_idx)

test_dataset = torch.utils.data.Subset(dataset, test_idx)

#Prepare training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Define k folds and hyper-parameters
n_folds = 5
n_epochs = [100]
learning_rates = [0.01]

gridsearch = {'epoch':n_epochs,
              'lr' : learning_rates}

#%%
#Cross Validation
final_results_dict = {}   


for n_epoch in gridsearch['epoch']:
    for lr in gridsearch['lr']:

        #Set fold
        kf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state=seed)
        
        #Set result outputs
        results_dict = {'Train Loss' : [], 'Train Accuracy' : [], 'Validation Loss' : [], 'Validation Accuracy' : []}
        
        #Loop through each fold for the dataset
        for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(train_dataset)), [label for _, label in train_dataset])):

            #Create train and val loader
            train_data = torch.utils.data.Subset(train_dataset, train_idx)
            val_data= torch.utils.data.Subset(train_dataset, val_idx)

            train_loader = DataLoader(dataset = train_data, batch_size=32, num_workers=0, pin_memory=True)
            val_loader = DataLoader(dataset = val_data, batch_size=32, num_workers=0, pin_memory=True)
            
            #Initialise model and optimiser
            train_model = ANNBasic(91, 3, n_hidden_array = [120], activationfunctions = [torch.nn.Sigmoid])
            
            optimiser = optim.SGD(train_model.parameters(), lr=lr, momentum=0.1)
            
            #Train
            train_loss, train_accuracy = train_batch(device, n_epoch, train_model, optimiser, train_loader)
            results_dict['Train Loss'].append(train_loss)
            results_dict['Train Accuracy'].append(train_accuracy)
            
            #Validation
            val_loss, val_accuracy = test(device, train_model, val_loader)
            results_dict['Validation Loss'].append(val_loss)
            results_dict['Validation Accuracy'].append(val_accuracy)
            
        #Append results
        config_label = "Seed:{}-Epoch:{}-LR:{}".format(seed, n_epoch, lr)
        
        #Calculate average results of all folds
        for key in results_dict.keys():
            results_dict[key] = np.mean(results_dict[key])
            
        final_results_dict[config_label] = results_dict
#%%
output_path = import_path + r'\outputs'
compute_results(output_path, final_results_dict, 'Training Output')



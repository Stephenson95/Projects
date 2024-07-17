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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#Reproducibility
seed = 0
torch.manual_seed(seed)

#Set import path
import_path = r"C:\Users\{}\Documents\GitHub\Projects\Insurance Cross Selling".format(os.getlogin())

#Import modules
sys.path.append(import_path + r'\modules')
from Dataloader import InsuranceDataset
from Model import ANN1layer
from utils import train_batch, test, compute_results, compute_probs

#Load dataset
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
scaler = StandardScaler()

dataset = InsuranceDataset(import_path + r'\data', 'train', transformer = scaler, encoder = encoder)

train_idx, test_idx = train_test_split(np.arange(len(dataset)),
                                             test_size=0.2,
                                             random_state=seed,
                                             shuffle=True,
                                             stratify=[label for _, label in dataset])

train_dataset = torch.utils.data.Subset(dataset, train_idx)

test_dataset = torch.utils.data.Subset(dataset, test_idx)

#Prepare training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
#Define k folds and hyper-parameters
n_folds = 10
n_epochs = [50, 100]
neurons = [22, 40, 80]
learning_rates = [0.1, 0.01]

gridsearch = {'epoch':n_epochs,
              'lr' : learning_rates,
              'neuron' : neurons}

#Cross Validation
final_results_dict = {}   

#Set fold
kf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state=seed)
kf_x = np.zeros(len(train_dataset))
kf_y = [label for _, label in train_dataset]

for n_epoch in gridsearch['epoch']:
    for neuron in gridsearch['neuron']:
        for lr in gridsearch['lr']:
    
            #Set result outputs
            results_dict = {'Train Loss' : [], 'Train Accuracy' : [], 'Validation Loss' : [], 'Validation Accuracy' : []}
            
            #Loop through each fold for the dataset
            for fold, (train_idx, val_idx) in enumerate(kf.split(kf_x, kf_y)):

                #Create train and val loader
                train_data = torch.utils.data.Subset(train_dataset, train_idx)
                val_data= torch.utils.data.Subset(train_dataset, val_idx)
                
                train_loader = DataLoader(dataset = train_data, batch_size=32, num_workers=0, pin_memory=True)
                val_loader = DataLoader(dataset = val_data, batch_size=32, num_workers=0, pin_memory=True)
                
                #Initialise model and optimiser
                train_model = ANN1layer(11, 2, neuron)                  
                
                optimiser = optim.Adam(train_model.parameters(), lr=lr)
                
                #Train
                train_loss, train_accuracy = train_batch(device, n_epoch, train_model, optimiser, train_loader)
                results_dict['Train Loss'].append(train_loss)
                results_dict['Train Accuracy'].append(train_accuracy)
                
                #Validation
                val_loss, val_accuracy = test(device, train_model, val_loader)
                results_dict['Validation Loss'].append(val_loss) 
                results_dict['Validation Accuracy'].append(val_accuracy)
                
            #Append results
            config_label = "Seed:{}-Epoch:{}-Neuron:{}".format(seed, n_epoch, neuron)
            
            #Calculate average results of all folds
            for key in results_dict.keys():
                results_dict[key] = np.mean(results_dict[key])
                
            final_results_dict[config_label] = results_dict

output_path = import_path + r'\outputs'
compute_results(output_path, final_results_dict, 'Training Output')

#%%
#Train Final model with Best hyper-parameters
n_epoch = 100
neuron = 200
n_layer = 1
lr = 0.1

#Reload train dataset
encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
scaler = StandardScaler()

train_loader = DataLoader(dataset = dataset, batch_size=32, num_workers=0, pin_memory=True)

final_model = ANN1layer(11, 2, neuron)
optimiser = optim.Adam(final_model.parameters(), lr=lr)
    
train_loss, train_accuracy = train_batch(device, n_epoch, final_model, optimiser, train_loader)
    
#%%
#Perform inference
import pandas as pd

#Load test dataset
test_encoder = dataset.return_encoder()
test_scaler = dataset.return_scaler()
testset = InsuranceDataset(import_path, 'test', transform=test_scaler, encoder=test_encoder)
test_loader = DataLoader(dataset = testset, batch_size=len(testset), num_workers=0, pin_memory=True)

final_pred = compute_probs(final_model, test_loader)

submissionfile = pd.concat([testset.return_ids(), pd.DataFrame(final_pred.cpu().numpy())], axis = 1)
submissionfile.rename(columns = {0:'Target'}, inplace=True)
submissionfile['Target'] = submissionfile['Target'].map({0:'Dropout', 1:'Enrolled', 2:'Graduate'})

submissionfile.to_csv(import_path + r'\outputs\submission_annbatch.csv', index=False)


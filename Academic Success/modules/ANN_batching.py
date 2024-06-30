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
import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
if (sklearn.__version__ == '1.1.2'):
    encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
else:
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
scaler = StandardScaler()
dataset = AcademicDataset(import_path, 'train', transform=scaler, encoder = encoder, return_onehotencoder=True)

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
n_epochs = [100]
neurons = [200]
learning_rates = [0.1]
layers = [1]

gridsearch = {'epoch':n_epochs,
              'lr' : learning_rates,
              'neuron' : neurons,
              'layers' : layers}

#Cross Validation
final_results_dict = {}   


for n_epoch in gridsearch['epoch']:
    for neuron in gridsearch['neuron']:
        for n_layer in gridsearch['layers']:
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
                    train_model = ANNBasic(257, 3, n_hidden_array = [round(neuron/(layer+1)) for layer in range(n_layer)], activationfunctions = [torch.nn.Sigmoid for layer in range(n_layer)])   

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
                config_label = "Seed:{}-Epoch:{}-Neuron:{}-Layers:{}".format(seed, n_epoch, neuron, n_layer)
                
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
if (sklearn.__version__ == '1.1.2'):
    encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
else:
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

scaler = StandardScaler()

dataset = AcademicDataset(import_path, 'train', transform=scaler, encoder=encoder, return_onehotencoder=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(dataset = dataset, batch_size=32, num_workers=0, pin_memory=True)

final_model = ANNBasic(257, 3, n_hidden_array = [round(neuron/(layer+1)) for layer in range(n_layer)], activationfunctions = [torch.nn.Sigmoid for layer in range(n_layer)])

optimiser = optim.Adam(final_model.parameters(), lr=lr)
    
train_loss, train_accuracy = train_batch(device, n_epoch, final_model, optimiser, train_loader)
    
print(r'Average training accuracy: {}'.format(train_accuracy))
#%%
#Perform inference
import pandas as pd

#Load test dataset
test_encoder = dataset.return_encoder()
scaler = dataset.return_scaler()
testset = AcademicDataset(import_path, 'test', transform=scaler, encoder=test_encoder, return_onehotencoder=False)
test_loader = DataLoader(dataset = testset, batch_size=len(testset), num_workers=0, pin_memory=True)

final_model.eval()

for data in test_loader:
    data = data.to(device)
    final_output = final_model(data)
    final_pred = final_output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

submissionfile = pd.concat([testset.return_ids(), pd.DataFrame(final_pred.cpu().numpy())], axis = 1)
submissionfile.rename(columns = {0:'Target'}, inplace=True)
submissionfile['Target'] = submissionfile['Target'].map({0:'Dropout', 1:'Enrolled', 2:'Graduate'})

submissionfile.to_csv(import_path + r'\outputs\submission_annbatch.csv', index=False)


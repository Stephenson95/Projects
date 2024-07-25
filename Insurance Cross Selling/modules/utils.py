# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:41:15 2024

@author: Stephenson
"""
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F

#Train function
def train_batch(gpu_config, epochs, model, optimiser, dataloader):
    model.to(gpu_config)
    model.train()
    loss_list = []
    accuracy_list = []
    for epoch in range(1, epochs + 1):
        correct = 0
        train_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):

            data, target = data.to(gpu_config), target.type(torch.LongTensor).to(gpu_config)
            model.zero_grad()
            optimiser.zero_grad()
            output = model(data)
            
            #Create weights for loss function
            #weights = target.unique(return_counts=True)[1].type(torch.float32).pow_(-1).to(gpu_config)
            #weights = weights/sum(weights)
            #loss = F.cross_entropy(output, target, weight = weights)
            loss = F.cross_entropy(output, target)
            
            train_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
            loss.backward()
            optimiser.step()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))
    
        train_loss /= len(dataloader)
        loss_list.append(train_loss)
        if len(dataloader) == 1:
            accuracy_list.append(100. * correct / len(data))
        else:
            accuracy_list.append(100. * correct / len(dataloader))

    return np.mean(loss_list), np.mean(accuracy_list)


#Test function
def test(gpu_config, model, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in dataloader:
        data, target = data.to(gpu_config), target.type(torch.LongTensor).to(gpu_config)
        output = model(data)
        
        test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))
    return test_loss, (100. * correct / len(dataloader.dataset)).item()


def compute_probs(gpu_config, model, dataloader):
    model.eval()
    for data in dataloader:
        data = data.to(gpu_config)
        output = model(data)
        pred = F.softmax(output, dim=1)[:,1].data
    return pred


#Compute results
def compute_results(output_path, validation_results, label):
    
    output = pd.DataFrame()
    
    #Loop over keys
    for i in validation_results.keys():
        temp_df = pd.DataFrame({k:v for k, v in validation_results[i].items()}, index = [i])
        output = pd.concat([output,temp_df], ignore_index=False)
        
    output.reset_index(inplace=True)
    output.sort_values(by = 'Validation Accuracy', ascending = True, inplace=True)
    output.to_csv(output_path + r'\{} Table by Seed.csv'.format(label), index=False)
    
    output['Config']  = output['index'].apply(lambda i : i[i.find("Epoch:"):])
    output = output.groupby(['Config']).agg({'Train Loss':'mean', 'Train Accuracy' : 'mean',
                                             'Validation Loss' : 'mean', 'Validation Accuracy' : 'mean'})
    
    output.sort_values(by = 'Validation Accuracy', ascending = True, inplace=True)
    
    output.sort_values(by = 'Validation Loss', ascending = True, inplace=True)
    output.to_csv(output_path + r'\{} Table.csv'.format(label), index=True)


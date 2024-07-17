# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:07:14 2024

@author: Stephenson
"""
import pandas as pd
import torch
from scipy.sparse import hstack
from torch.utils.data import Dataset

#Customized Dataset
class InsuranceDataset(Dataset):
    def __init__(self, import_path, file_name, transformer = None, encoder = None):
        
        #Load data
        train_data = pd.read_csv(import_path + f'\{file_name}.csv')
        output_id = train_data['id']
        train_data.drop('id', axis = 'columns', inplace=True)
        
        #Split labels and data
        if file_name == 'train':
            X = train_data.iloc[:,:-1]
            y = train_data.iloc[:,-1]
        else:
            X = train_data
        
        #Categorise Columns
        cat_cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
        num_cols = list(set(X.columns) - set(cat_cols))
        
        #One hot encode
        if file_name == 'train':
            temp_data_enc = encoder.fit_transform(X[cat_cols])
        else:
            temp_data_enc = encoder.transform(X[cat_cols])
            
        #Standardise
        if file_name == 'train':
            temp_data_std = transformer.fit_transform(X[num_cols])
        else:
            temp_data_std = transformer.transform(X[num_cols])
            
        X = hstack([temp_data_std, temp_data_enc])
            
            
        #Convert to tensor
        X = torch.tensor(X.toarray(), dtype = torch.float32)

        self.file_name = file_name
        self.x = X
        if self.file_name == 'train':
            y = torch.tensor(y, dtype = torch.float32)
            self.y = y
        self.n_samples = len(self.x)
        self.transformer = transformer
        self.encoder = encoder
        self.output_id = output_id

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.file_name == 'train':
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]
        
    def return_encoder(self):
        return self.encoder

    def return_scaler(self):
        return self.transformer
        
    def return_ids(self):
        return self.output_id





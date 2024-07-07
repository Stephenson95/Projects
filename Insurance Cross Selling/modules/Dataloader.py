# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:07:14 2024

@author: Stephenson
"""
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

#Customized Dataset
class AcademicDataset(Dataset):
    def __init__(self, import_path, file_name, transform = None, encoder = None, return_onehotencoder = False):
        
        #Load data
        train_data = pd.read_csv(import_path + r'\{}.csv'.format(file_name))
        output_id = train_data['id']
        train_data.drop(['id', 'Nacionality'], axis = 'columns', inplace=True)
        
        #Remove unwanted columns
        #cols_to_drop = ["Mother's occupation", "Father's occupation", "Mother's qualification", "Father's qualification", "Nacionality"]
        #train_data.drop(cols_to_drop, axis = 'columns', inplace=True)
        
        #Split labels and data
        if file_name == 'train':
            X = train_data.iloc[:,0:-1]
            y = train_data.iloc[:,-1]
        else:
            X = train_data.iloc[:,:]
        
        #Define category columns to be one-hot encoded
        cat_cols = ['Marital status', 'Application mode', 'Course', 'Previous qualification', "Mother's occupation", "Father's occupation", "Mother's qualification", "Father's qualification"]
        
        if return_onehotencoder:
            temp_data = encoder.fit_transform(X[cat_cols])
        else:
            temp_data = encoder.transform(X[cat_cols])
            
            
        temp_X = pd.DataFrame(temp_data, columns=encoder.get_feature_names_out(cat_cols))
        X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True), temp_X.reset_index(drop=True)], axis = 1)
        
        if file_name == 'train':
            #Encode label
            le = LabelEncoder()
            y = le.fit_transform(y)
            y = torch.tensor(y, dtype = torch.float32)
            
        if transform:
            X = transform.fit_transform(X)
            
        #Convert to tensor
        X = torch.tensor(X, dtype = torch.float32)
            
        self.x = X
        self.file_name = file_name
        if self.file_name == 'train':
            self.y = y
        self.n_samples = len(self.x)
        self.transform = transform
        self.return_onehotencoder = return_onehotencoder
        self.onehotencoder = encoder
        self.output_id = output_id

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.file_name == 'train':
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]
        
    def return_encoder(self):
        if self.return_onehotencoder:
            return self.onehotencoder
        else:
            return None
        
    def return_scaler(self):
        return self.transform
        
    def return_ids(self):
        return self.output_id





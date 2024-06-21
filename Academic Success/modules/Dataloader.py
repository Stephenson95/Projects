# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:07:14 2024

@author: Stephenson
"""
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torch.utils.data import Dataset

#Customized Dataset
class AcademicDataset(Dataset):
    def __init__(self, import_path, transform = None):
        
        #Load data
        train_data = pd.read_csv(import_path + r'\train.csv')
        train_data.drop('id', axis = 'columns', inplace=True)
        
        #Remove unwanted columns
        cols_to_drop = ["Mother's occupation", "Father's occupation", "Mother's qualification", "Father's qualification", "Nacionality"]
        train_data.drop(cols_to_drop, axis = 'columns', inplace=True)
        
        #Split labels and data
        X = train_data.iloc[:,0:-1]
        y = train_data.iloc[:,-1]
        
        #Define category columns to be one-hot encoded
        cat_cols = ['Marital status', 'Application mode', 'Course', 'Previous qualification']
        enc = OneHotEncoder(sparse_output=False , drop='first')
        
        temp_data = enc.fit_transform(X[cat_cols])
        temp_X = pd.DataFrame(temp_data, columns=enc.get_feature_names_out(cat_cols))
        X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True), temp_X.reset_index(drop=True)], axis = 1)
        
        #Encode label
        le = LabelEncoder()
        le.fit(['Dropout', 'Enrolled', 'Graduate'])
        y = le.transform(y)
        
        if transform:
            X = transform.fit_transform(X)
            
        #Convert to tensor
        X = torch.tensor(X, dtype = torch.float32)
        y = torch.tensor(y, dtype = torch.float32)
            
        self.x = X
        self.y = y
        self.n_samples = len(y)
        self.transform = transform

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    


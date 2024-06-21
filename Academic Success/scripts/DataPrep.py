# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:53:23 2024

@author: Stephenson
"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA

def load_data(import_path, train_ratio, normalisation = True):

    train_data = pd.read_csv(import_path + r'\train.csv')
    train_data.drop('id', axis = 'columns', inplace=True)
    
    cols_to_drop = ["Mother's occupation", "Father's occupation", "Mother's qualification", "Father's qualification", "Nacionality"]
    train_data.drop(cols_to_drop, axis = 'columns', inplace=True)
    
    X = train_data.iloc[:,0:-1]
    y = train_data.iloc[:,-1]
    
    #Define category columns to be one-hot encoded
    cat_cols = ['Marital status', 'Application mode', 'Course', 'Previous qualification']
    enc = OneHotEncoder(sparse_output=False , drop='first')
    
    temp_data = enc.fit_transform(X[cat_cols])
    temp_X = pd.DataFrame(temp_data, columns=enc.get_feature_names_out(cat_cols))
    X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True), temp_X.reset_index(drop=True)], axis = 1)
    
    #Split dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                         test_size=train_ratio,
                                                         random_state=10,
                                                         shuffle=True,
                                                         stratify=y)
    
    ##Label encoder
    le = LabelEncoder()
    le.fit(['Dropout', 'Enrolled', 'Graduate'])
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    
    ##Retain columns
    X_cols = x_train.columns.tolist()
    
    ##Standardise features
    scaler = MinMaxScaler()
    X_scaled_train = scaler.fit_transform(x_train)
    X_scaled_test = scaler.fit_transform(x_test)
    
    ##Perform PCA
    pca_cols = ['Curricular units 1st sem (credited)',
                'Curricular units 1st sem (enrolled)',
                'Curricular units 1st sem (evaluations)',
                'Curricular units 1st sem (approved)',
                'Curricular units 1st sem (grade)',
                'Curricular units 1st sem (without evaluations)',
                'Curricular units 2nd sem (credited)',
                'Curricular units 2nd sem (enrolled)',
                'Curricular units 2nd sem (evaluations)',
                'Curricular units 2nd sem (approved)',
                'Curricular units 2nd sem (grade)',
                'Curricular units 2nd sem (without evaluations)']
    
    for i in range(int(len(pca_cols)/2)):
        
        pca = PCA(n_components = 1)
        col_ind = [X_cols.index(pca_cols[i]), X_cols.index(pca_cols[i+int(len(pca_cols)/2)])]
        principal_components_train = pca.fit_transform(X_scaled_train[:, col_ind])
        principal_components_test = pca.fit_transform(X_scaled_test[:, col_ind])
        
        #Drop columns from previous ndarray and column list
        X_scaled_train = np.delete(X_scaled_train, col_ind, axis = 1)
        X_scaled_test = np.delete(X_scaled_test, col_ind, axis = 1)
        X_cols = [X_col for X_col in X_cols if X_col not in [pca_cols[i], pca_cols[i+int(len(pca_cols)/2)]]]
        
        #Add pca component to ndarray and column list
        X_scaled_train = np.append(X_scaled_train, principal_components_train, axis = 1)
        X_scaled_test = np.append(X_scaled_test, principal_components_test, axis = 1)
        X_cols.append(pca_cols[i].replace("1st sem ", ""))
    
    train_final = np.append(X_scaled_train, y_train.reshape(-1,1), axis = 1)
    test_final = np.append(X_scaled_test, y_test.reshape(-1,1), axis = 1)
    
    return X_cols, train_final, test_final

#%%
import_path = r'C:\Users\Stephenson\Documents\GitHub\Projects\Academic Success'
X_cols, train_final, test_final =  load_data(import_path, 0.2)

train_final

pd.DataFrame(train_final, columns = X_cols + ['Target']).to_csv(import_path + r'\processed_data\train_data.csv', index=False)
pd.DataFrame(test_final, columns = X_cols + ['Target']).to_csv(import_path + r'\processed_data\test_data.csv', index=False)

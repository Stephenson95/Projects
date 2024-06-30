# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:31:39 2024

@author: 61450
"""

#Import standard libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from catboost import CatBoostClassifier

#Reproducibility
seed = 0

#Set import path
import_path = r"C:\Users\{}\Documents\GitHub\Projects\Academic Success".format(os.getlogin())

#Load dataset
train_data = pd.read_csv(import_path + r'\{}.csv'.format('train'))
train_data.drop(['id', 'Nacionality'], axis = 'columns', inplace=True)
cat_features = [train_data.columns.get_loc(c) for c in ['Marital status', 'Application mode', 'Course', 'Previous qualification', "Mother's occupation", "Father's occupation", "Mother's qualification", "Father's qualification"] if c in train_data]

#Split labels and data
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

#Encode   
le = LabelEncoder()
y = le.fit_transform(y)

#Split
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                        test_size=0.2,
                                        random_state=seed,
                                        shuffle=True)

#%%
#Base Model

final_model = CatBoostClassifier()
final_model.fit(x_train,y_train, cat_features=cat_features)
y_pred = final_model.predict(x_test)


test_data = pd.read_csv(import_path + r'\{}.csv'.format('test'))
output_id = test_data['id']
test_data.drop(['id', 'Nacionality'], axis = 'columns', inplace=True)

#Standardise
y_pred = final_model.predict(test_data)

output = le.inverse_transform(y_pred)
submissionfile = pd.concat([pd.Series(output_id), pd.Series(output)], axis = 1)
submissionfile.rename(columns = {0:'Target'}, inplace=True)
submissionfile.to_csv(import_path + r'\outputs\submission_cat.csv', index=False)
 

#%%
#Cross Validation
best_params = None
best_score = 0
final_results_dict = {}   


#Define hyper-parameters

param_grid={"learning_rate": [0.01, 0.03, 0.05],
            "iterations": [200, 500, 1000],
            "depth" : [4, 6, 8, 10],
            "l2_leaf_reg": [1, 3, 10]} 

#Set fold
kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state=seed)

for learning_rate in param_grid['learning_rate']:
    for iterations in param_grid['iterations']:
        for depth in param_grid['depth']:
            for l2_leaf_reg in param_grid['l2_leaf_reg']:
            
                test_scores = []
                best_rounds = []
                
                for fold, (train_index, val_index) in enumerate(kf.split(x_train, y_train)):
                    
                    X_train_fold, X_val_fold = x_train.iloc[train_index,:], x_train.iloc[val_index, :]
                    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        
                    # Prepare the model
                    model = CatBoostClassifier(learning_rate=learning_rate,
                                                iterations=iterations,
                                                depth = depth,
                                                l2_leaf_reg = l2_leaf_reg,
                                                random_state=seed)
        
                    # Fit model on train fold and use validation for early stopping
                    model.fit(X_train_fold, y_train_fold, cat_features=cat_features)
        
                    # Predict on test set
                    y_pred_test = model.predict(X_val_fold)
                    test_score = accuracy_score(y_val_fold, y_pred_test)
                    test_scores.append(test_score)
                    
                    print(r"Fold {}: Accuracy {}".format(fold, test_score))
        
                # Compute average score across all folds
                average_score = np.mean(test_scores)
                if average_score > best_score:
                    best_score = average_score
                    best_params = {'learning_rate': [learning_rate], 
                                   'iterations': [iterations],
                                   'depth': [depth],
                                   'l2_leaf_reg' : [l2_leaf_reg]}

print(f"Best Parameters: {best_params}")
print(f"Best CV Average Accuracy: {best_score}")

np.save(import_path + r'/outputs/cat_tuned_params.npy', best_params) 
#%%
#Test on hold out set
best_params = np.load(import_path + r'/outputs/lgbm_tuned_params.npy', allow_pickle='TRUE').item()

for learning_rate in param_grid['learning_rate']:
    for iterations in param_grid['iterations']:
        for depth in param_grid['depth']:
            for l2_leaf_reg in param_grid['l2_leaf_reg']:
                
                final_model = CatBoostClassifier(learning_rate=learning_rate,
                                            iterations=iterations,
                                            depth = depth,
                                            l2_leaf_reg = l2_leaf_reg,
                                            random_state=seed)
                
                # Fit model on train fold and use validation for early stopping
                final_model.fit(x_train, y_train, cat_features=cat_features)
            
   
test_score = accuracy_score(y_test, final_model.predict(x_test))
print(test_score)
  
        
#%%
#Retrain entire dataset on best parameters
for learning_rate in param_grid['learning_rate']:
    for iterations in param_grid['iterations']:
        for depth in param_grid['depth']:
            for l2_leaf_reg in param_grid['l2_leaf_reg']:
                    
                final_model = CatBoostClassifier(learning_rate=learning_rate,
                                            iterations=iterations,
                                            depth = depth,
                                            l2_leaf_reg = l2_leaf_reg,
                                            random_state=seed)
                
                # Fit model on train fold and use validation for early stopping
                final_model.fit(X, y, cat_features=cat_features)

#%%
#Prediction on test set
test_data = pd.read_csv(import_path + r'\{}.csv'.format('test'))
output_id = test_data['id']
test_data.drop(['id', 'Nacionality'], axis = 'columns', inplace=True)

#Remove unwanted columns
#cols_to_drop = ["Mother's occupation", "Father's occupation", "Mother's qualification", "Father's qualification", "Nacionality"]
#test_data.drop(cols_to_drop, axis = 'columns', inplace=True)

#Make prediction (using number of trees based on trained stoping round) 
final_pred = final_model.predict(test_data) 

#Export  
output = le.inverse_transform(final_pred)
submissionfile = pd.concat([pd.Series(output_id), pd.Series(output)], axis = 1)
submissionfile.rename(columns = {0:'Target'}, inplace=True)

submissionfile.to_csv(import_path + r'\outputs\submission_cat.csv', index=False)
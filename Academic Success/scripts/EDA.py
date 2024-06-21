# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:53:23 2024

@author: Stephenson
"""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
import xgboost as xgb

import_path = r'C:\Users\Stephenson\Documents\GitHub\Projects\Academic Success'

train_data = pd.read_csv(import_path + r'\train.csv')

train_data.drop('id', axis = 'columns', inplace=True)
col_list = train_data.columns.tolist()

cols_to_drop = ["Mother's occupation", "Father's occupation", "Mother's qualification", "Father's qualification", "Nacionality"]
train_data.drop(cols_to_drop, axis = 'columns', inplace=True)

#Split dataset




X = train_data.iloc[:,0:-1]
y = train_data.iloc[:,-1]

##Label encoder
le = LabelEncoder()
le.fit(['Dropout', 'Enrolled', 'Graduate'])
y = le.transform(y)

##One hot encode
cat_cols = ['Marital status', 'Application mode', 'Course', 'Previous qualification']
enc = OneHotEncoder(sparse_output=False , drop='first')
temp_data = enc.fit_transform(X[cat_cols])
temp_X = pd.DataFrame(temp_data, columns=enc.get_feature_names_out(cat_cols))
X = pd.concat([X.drop(columns=cat_cols), temp_X], axis = 1)

##Retain columns
X_cols = X.columns.tolist()

##Standardise features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

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
    principal_components = pca.fit_transform(X_scaled[:, col_ind])
    
    #Drop columns from previous ndarray and column list
    X_scaled = np.delete(X_scaled, col_ind, axis = 1)
    X_cols = [X_col for X_col in X_cols if X_col not in [pca_cols[i], pca_cols[i+int(len(pca_cols)/2)]]]
    
    #Add pca component to ndarray and column list
    X_scaled = np.append(X_scaled, principal_components, axis = 1)
    X_cols.append(pca_cols[i].replace("1st sem ", ""))

pd.DataFrame(X_scaled, columns = X_cols).to_csv(import_path + r'\processed_data\train_data.csv')



#%%
BestFeatures = SelectKBest(score_func=chi2, k=10)

fit = BestFeatures.fit(X_scaled,y)

df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
f_Scores = pd.concat([df_columns,df_scores],axis=1)
f_Scores.columns = ['Specs','Score']  
f_Scores.sort_values(by = 'Score', inplace=True, ascending=False)
f_Scores.reset_index(inplace=True, drop=True)

f_Scores.plot()

#%%
model = xgb.XGBClassifier()
model.fit(X_scaled,y)
feat_imp = pd.Series(model.feature_importances_, index=X.columns)
feat_imp.nlargest(10).plot(kind='barh')

plt.figure(figsize=(8,6))
plt.show()

#%%

cormat = X.corr()
top_corr_features = cormat.index

plt.figure(figsize=(80,80))

#plot heat map
g=sns.heatmap(X[top_corr_features].corr(),annot=True,cmap="RdYlGn")
  
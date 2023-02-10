# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 22:47:14 2023

@author: Stephenson
"""
import pandas as pd
import numpy as np
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
pd.options.mode.chained_assignment = None

device = "cuda" if torch.cuda.is_available() else "cpu"

#Import
file_path = r'C:\Users\Stephenson\Desktop\Code\ASD'
#file_path = r'C:\Users\61450\Documents\Python Scripts\ASD'
column_headers = pd.read_excel(file_path + r'\CSV.header.fieldids.xlsx', sheet_name = 'Sheet1')

data = pd.DataFrame(columns = column_headers.iloc[:,0])
for file in glob.glob(file_path + r'\*.csv'):
    data = pd.concat([data, pd.read_csv(file, names = column_headers.iloc[:,0], delimiter = r'\t', engine = 'python')], axis = 0)

#Clean
#Format dates and remove unrequired fields
data['SQLDATE'] = pd.to_datetime(data['SQLDATE'], format = '%Y%m%d')
data.drop(['GLOBALEVENTID', 'MonthYear'], axis = 'columns', inplace=True)

#Ensure no blanks in goldsteinscale and country fields
data = data[~pd.isna(data['GoldsteinScale'])]
data = data[~(pd.isna(data['ActionGeo_Lat']) | pd.isna(data['ActionGeo_Long']))]

#Drop duplicates
data = data.drop_duplicates() #incase there are duplicate events

data.reset_index(drop=True, inplace=True)

#Formatting and subset of interested data types
col_to_take = ['ActionGeo_CountryCode', 'GoldsteinScale', 'QuadClass', 'NumMentions', 'NumSources', 'NumArticles', 'AvgTone', 'ActionGeo_Lat', 'ActionGeo_Long']
data_final = data[col_to_take]
data_final['QuadClass'] = pd.Categorical(data_final['QuadClass'], ordered = None)
data_final['NumMentions'] = data_final['NumMentions'].astype(int)
data_final['NumSources'] = data_final['NumSources'].astype(int)
data_final['NumArticles'] = data_final['NumArticles'].astype(int)

#PCA
SS_scaler = StandardScaler()
data_final['NumArticles_scaled'] = SS_scaler.fit_transform(data_final[['NumArticles']])
data_final['NumMentions_scaled'] = SS_scaler.fit_transform(data_final[['NumMentions']])
data_final['NumSources_scaled'] = SS_scaler.fit_transform(data_final[['NumSources']])

pca = PCA(n_components=1)

pca.fit(data_final[['NumArticles_scaled', 'NumMentions_scaled', 'NumSources_scaled']])

print(pca.explained_variance_ratio_) #First component explains 89% of variance

data_final['PCA_publicity'] = pca.components_[0][0] * data_final['NumArticles_scaled'] +  pca.components_[0][1] * data_final['NumMentions_scaled'] +  pca.components_[0][2] * data_final['NumSources_scaled']

#Create classification dataset
ord_data_final = data_final.copy()

bins = pd.IntervalIndex.from_tuples([(-10, -5), (-5, 0), (0, 5), (5, 10.1)], closed = 'left')
labels = ['Major Conflict', 'Minor Conflict', 'Minor Cooperation', 'Major Cooperation']
ord_data_final['GoldsteinScale'] = pd.cut(ord_data_final['GoldsteinScale'], bins = bins, right = False, include_lowest = True).map(dict(zip(bins, labels)))

ord_data_final = pd.get_dummies(ord_data_final, columns =['QuadClass'], drop_first=True)     

ord_data_final['GoldsteinScale'] = ord_data_final['GoldsteinScale'].map(dict(zip(labels, [0,1,2,3])))

X, y = ord_data_final.loc[:, ['QuadClass_2', 'QuadClass_3', 'QuadClass_4', 'AvgTone', 'PCA_publicity']], ord_data_final.loc[:, "GoldsteinScale"]

X = torch.tensor(X.values).type(torch.float)

y = torch.tensor(y.values).type(torch.float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


#Modeling
class ClassifierV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=5, out_features = 16)
        self.layer_2 = nn.Linear(in_features=16, out_features = 4)
        
    def forward(self, x):
        return self.layer_2(self.layer_1(x))

class ClassifierV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=5, out_features = 16)
        self.layer_2 = nn.Linear(in_features=16, out_features = 16)
        self.layer_3 = nn.Linear(in_features=16, out_features = 4)
        
    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))

class ClassifierV3(nn.Module):
    def __init__ (self, input_features, output_features, hidden_units = 8):
        """
        Initialises multiclass classification model
        
        Parameters
        ----------
        input_features : int
            DESCRIPTION. Number of input features to the model
        output_features : int
            DESCRIPTION. Number of output feautres (number of output classes)
        hidden_units : int, optional
            DESCRIPTION. Number of hidden units between layers. The default is 8.

        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
            )
    def forward(self, x):
        return self.linear_layer_stack(x)
    
     
model_1 = ClassifierV1().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

# model_1.eval()
# with torch.inference_mode():
#     y_logits = model_1(X_test.to(device))[:5]
# y_logits

# y_pred_probs = torch.sigmoid(y_logits)
# y_pred_probs

# torch.argmax(y_pred_probs, dim=1)

#Build training and testing loop
torch.cuda.manual_seed(42)
epochs = 5000

y_train = y_train.type(torch.LongTensor)
y_test = y_test.type(torch.LongTensor)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


for epoch in range(epochs):
    model_1.train()
    
    y_logits = model_1(X_train).squeeze()
    y_pred = torch.argmax(torch.sigmoid(y_logits), dim=1)
    
    loss = loss_fn(y_logits, y_train)
    
    acc = accuracy_fn(y_true = y_train,
                      y_pred = y_pred)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.argmax(torch.sigmoid(test_logits), dim=1)
        
        test_loss = loss_fn(test_logits,y_test)
        
        test_acc = accuracy_fn(y_true = y_test, 
                               y_pred = test_pred)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
        
#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(ord_data_final[['PCA_publicity']])
ord_data_final['PCA_publicity'] = scaler.transform(ord_data_final[['PCA_publicity']])


model_2 = ClassifierV3(input_features = 5, output_features = 4, hidden_units = 8).to(device)

#Build training and testing loop
torch.cuda.manual_seed(42)
epochs = 5000

y_train = y_train.type(torch.LongTensor)
y_test = y_test.type(torch.LongTensor)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

for epoch in range(epochs):
    model_2.train()
    
    y_logits = model_2(X_train)
    y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
    
    loss = loss_fn(y_logits, y_train)
    
    acc = accuracy_fn(y_true = y_train,
                      y_pred = y_pred)
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    model_2.eval()
    with torch.inference_mode():
        test_logits = model_2(X_test)
        test_pred = torch.argmax(torch.softmax(test_logits, dim=1), dim=1)
        
        test_loss = loss_fn(test_logits,y_test)
        
        test_acc = accuracy_fn(y_true = y_test, 
                               y_pred = test_pred)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
        
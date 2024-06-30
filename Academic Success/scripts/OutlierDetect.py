# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 23:25:34 2024

@author: 61450
"""
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.manifold import TSNE

#Reproducibility
seed = 0

#Set import path
import_path = r"C:\Users\{}\Documents\GitHub\Projects\Academic Success".format(os.getlogin())

#Load dataset
train_data = pd.read_csv(import_path + r'\{}.csv'.format('train'))
train_data.drop(['id', 'Nacionality'], axis = 'columns', inplace=True)

#Split labels and data
X = train_data.iloc[:,:-1]
y = train_data.iloc[:,-1]

#Encode   
le = LabelEncoder()
y = le.fit_transform(y)

#Standardise
scaler = StandardScaler()
X = scaler.fit_transform(X)

#TSNE model
X_embedded = TSNE(n_components=3, learning_rate='auto',
                  init='random', perplexity=10, random_state=seed).fit_transform(X)


df_subset = pd.DataFrame()
df_subset['tsne-2d-one'] = X_embedded[:,0]
df_subset['tsne-2d-two'] = X_embedded[:,1]
df_subset['tsne-2d-three'] = X_embedded[:,2]
df_subset['y'] = y

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 3),
    data=df_subset,
    legend="full",
    alpha=0.3
)

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-three",
    hue="y",
    palette=sns.color_palette("hls", 3),
    data=df_subset,
    legend="full",
    alpha=0.3
)

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-three", y="tsne-2d-one",
    hue="y",
    palette=sns.color_palette("hls", 3),
    data=df_subset,
    legend="full",
    alpha=0.3
)
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 21:31:44 2022

@author: Stephenson
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #This is for 3d scatter plots.
import math
import random
import functools
from scipy.stats import multivariate_normal

X = np.load(r"C:\Users\Stephenson\Desktop\ANU\Introduction to Machine Learning\Assignment\Assignment 3\A3_programming\A3_programming\data.npy")
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:, 2])
plt.show()

def initialise_parameters(X, K):
    # YOUR CODE HERE
    pi = np.divide(np.ones(K), K)
    mu = X[np.random.choice(X.shape[0], K, replace=False)]
    sigma = []
    for i in range(K):
        temp = np.random.rand(X.shape[1],X.shape[1])
        sigma.append(np.dot(temp, temp.T))
    sigma = np.stack(sigma, axis = 0)
    return sigma, mu, pi


def E_step(pi, mu, sigma, X):
    #Intialise output
    output = np.empty((X.shape[0], len(pi)))
    #Calculate probabilities for each cluster
    for idx, mean in enumerate(mu):
        output[:, idx] = multivariate_normal.pdf(X, mean, sigma[idx])
    #Normalise
    for idx in range(output.shape[0]):
        denom = np.dot(pi, output[idx, :])
        for i in range(output.shape[1]):
            output[idx, i] = pi[i]*output[idx, i]/denom
    return output


K = 4
sigma, mu, pi = initialise_parameters(X[:, :3], K)
responsibilities = E_step(pi, mu, sigma, X[:, :3])

def M_step(r, X):
    #Update mean
    mu = []
    for idx in range(r.shape[1]):
        mu.append(np.dot(X.T, r[:, idx])/np.sum(r[:, idx]))
    mu = np.stack(mu)
    #Update sigma
    sigma = []
    for idx in range(r.shape[1]):
        temp = np.dot(r[:, idx]*(X - mu[idx]).T, (X-mu[idx]))/np.sum(r[:, idx])
        sigma.append(temp)
    sigma = np.stack(sigma, axis = 0)
    #Update pi
    pi = np.empty(r.shape[1])
    for idx in range(r.shape[1]):
        pi[idx] = np.sum(r[:, idx])/r.shape[0]
    return mu, sigma, pi


def EM(X, K, iterations):
    #Initialise parameters
    sigma, mu, pi = initialise_parameters(X, K)
    for i in range(iterations):
        r = E_step(pi, mu, sigma, X)
        mu, sigma, pi = M_step(r, X)
    return mu, sigma, pi

def classify(pi, mu, sigma, x):
    global output_dict
    output_dict = dict()
    for i in range(len(pi)):
        output_dict[i] = pi[i]*multivariate_normal.pdf(x, mu[i], sigma[i])
    
    return max(output_dict, key=output_dict.get)

def allocator(pi, mu, sigma, X, k):
    N = X.shape[0]
    cluster = []
    for ix in range(N):
        prospective_k = classify(pi, mu, sigma, X[ix, :])
        if prospective_k == k:
            cluster.append(X[ix, :])
    return np.asarray(cluster)

K = 5
iterations = 10
image = plt.imread(r"C:\Users\Stephenson\Desktop\ANU\Introduction to Machine Learning\Assignment\Assignment 3\A3_programming\A3_programming\mandm.png")


feature_matrix = np.reshape(image, (-1,image.shape[2]))
#EM algorithm
mu, sigma, pi = EM(feature_matrix, K, iterations)
#Prepare output matrix based on classification of each pixel
output_matrix = np.empty(feature_matrix.shape[0])
for i in range(feature_matrix.shape[0]):
    output_matrix[i] = classify(pi, mu, sigma, feature_matrix[i,:])
#Return reshaped output as original image matrix
output_matrix = np.reshape(output_matrix, (image.shape[0], image.shape[1]))
    

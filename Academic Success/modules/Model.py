# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:39:41 2024

@author: Stephenson
"""
import torch
import torch.nn as nn

#ANN Network
class ANNBasic(nn.Module):
    def __init__(self, n_inputs, n_classes, n_hidden_array = [], activationfunctions = []):
        super(ANNBasic, self).__init__()
        self.layers= []
        self.activationfunctions = [x() for x in activationfunctions]
        self.hidden_layers = len(n_hidden_array)
        for i in range(self.hidden_layers):
            self.layers.append(torch.nn.Linear(n_inputs, n_hidden_array[i]))
            n_inputs = n_hidden_array[i]
        self.layers = torch.nn.ModuleList(self.layers)
        self.output = torch.nn.Linear(n_inputs, n_classes)
    def forward(self, x):
        out = x
        for i in range(self.hidden_layers):
            out = self.layers[i](out)
            out = self.activationfunctions[i](out)
        y_pred = self.output(out)
        return y_pred


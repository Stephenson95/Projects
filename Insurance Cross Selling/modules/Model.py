# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:39:41 2024

@author: Stephenson
"""
import torch
import torch.nn as nn

#ANN Network
class ANN1layer(nn.Module):
    def __init__(self, n_inputs, n_classes, neurons):
        super(ANN1layer, self).__init__()
        self.layer1 = torch.nn.Linear(n_inputs, neurons)
        self.output = torch.nn.Linear(neurons, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.sigmoid(out)

        y_pred = self.output(out)
        return y_pred
    

class ANN2layer(nn.Module):
    def __init__(self, n_inputs, n_classes, neurons):
        super(ANN2layer, self).__init__()
        self.layer1 = torch.nn.Linear(n_inputs, neurons)
        self.layer2 = torch.nn.Linear(neurons, round(neurons/2))
        self.output = torch.nn.Linear(round(neurons/2), n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.tanh(out)
        
        out = self.layer2(out)
        out = torch.tanh(out)
        
        y_pred = self.output(out)
        return y_pred
    

class ANN3layer(nn.Module):
    def __init__(self, n_inputs, n_classes, neurons):
        super(ANN3layer, self).__init__()
        self.layer1 = torch.nn.Linear(n_inputs, neurons)
        self.layer2 = torch.nn.Linear(neurons, round(neurons/2))
        self.layer3 = torch.nn.Linear(round(neurons/2), round(neurons/4))
        self.output = torch.nn.Linear(round(neurons/4), n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = torch.tanh(out)
        
        out = self.layer2(out)
        out = torch.tanh(out)
        
        out = self.layer3(out)
        out = torch.tanh(out)
        
        y_pred = self.output(out)
        return y_pred


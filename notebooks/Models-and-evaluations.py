# python file for defining models and evaluation metrics for the autoregressive project
# Pretty notebooks and good hpc practices, yay:D 

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Setting up variables, these variables are weights in your 
        # network that can be updated while running our graph.
        # Notice, to make a hidden layer, the weights need to have the 
        # following dimensionality:
        #   W[number_of_units_going_out, number_of_units_going_in]
        #   b[number_of_units_going_out]
        # in the example below we have 2 input units (num_features) and 2 output units (num_output)
        # so our weights become W[2, 2], b[2]
        # if we want to make a hidden layer with 100 units, we need to define the shape of the
        # first weight to W[100, 2], b[2] and the shape of the second weight to W[2, 100], b[2]
        
        # first layer
        #self.W_1 = nn.Parameter(torch.randn(num_output, num_features)) 
        #self.b_1 = nn.Parameter(torch.randn(num_output))
        self.W_1 = nn.Parameter(torch.randn(hidden_layer_nodes, num_features)) 
        self.b_1 = nn.Parameter(torch.randn(hidden_layer_nodes))
                
        # second layer (to be completed as an exercise)
        # NB when you create a second layer, remember that you also must change parts of the first layer
        self.W_2 = nn.Parameter(torch.randn(hidden_layer_nodes, hidden_layer_nodes))
        self.b_2 = nn.Parameter(torch.randn(hidden_layer_nodes))

        self.W_3 = nn.Parameter(torch.randn(num_output, hidden_layer_nodes))
        self.b_3 = nn.Parameter(torch.randn(num_output))
        
    def forward(self, x):
        # Setting up ops, these ops will define edges along our computational graph
        # The below ops will compute a logistic regression, 
        # but can be modified to compute a neural network
        x = F.linear(x, self.W_1, self.b_1)
        
        # second layer (to be completed as an exercise)
        # NB when you create a second layer, remember that you also must change parts of the first layer
        #x = F.linear(x, self.W_2, self.b_2)
        x = F.relu(x)
        x = F.linear(x, self.W_2, self.b_2)

        x = F.relu(x)
        x = F.linear(x, self.W_3, self.b_3)

        return F.softmax(x, dim=1) # softmax to be performed on the second dimension

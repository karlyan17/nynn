#!/bin/python

import numpy as np

def sigma(x,deriv=False):
    #sigma function
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

class network:
    def __init__(self,len_inputs,n_inputs,n_outputs,depth_nn):
	self.n_inputs = n_inputs
	self.len_inputs = len_inputs
	self.n_outputs = n_outputs
	self.depth = depth_nn
	self.hidden_weights = [None] * depth_nn
	self.data = [None] * (depth_nn + 1)
	self.hidden_biases = [None] * depth_nn
	self.learning_rate = 0.5
	self.output = np.zeros((self.len_inputs,n_outputs))
	for i in range(self.depth):
            self.hidden_weights[i] = np.random.random((n_inputs,n_inputs))
	    self.hidden_biases[i] = np.random.random((len_inputs,n_inputs))
	self.output_weights = np.random.random((n_inputs,n_outputs))
	self.output_biases = np.random.random((len_inputs,n_outputs))
    
    
    def forward(self, nn_inputs):
	self.data[0] = nn_inputs
	for i in range(self.depth):
	    self.data[i+1] = sigma(np.dot(self.data[i],self.hidden_weights[i]) + self.hidden_biases[i])
	self.output = sigma(np.dot(self.data[-1],self.output_weights) + self.output_biases)
	return self.output
    
    def backprop(self, target_outputs):
        final_error = target_outputs - self.output
	running_error = final_error
	delta_out = running_error * sigma(self.output,deriv=True)
	running_error = np.dot(delta_out,self.output_weights.T)
	self.output_weights += self.learning_rate * np.dot(self.data[-1].T,delta_out)
	self.output_biases += self.learning_rate * delta_out
	for i in range(self.depth):
	    delta = running_error * sigma(self.data[-1-i],deriv=True)
	    running_error = np.dot(delta,self.hidden_weights[-1-i].T)
	    self.hidden_weights[-1-i] += self.learning_rate * np.dot(self.data[-1-i].T,delta)
	    self.hidden_biases[-1-i] += self.learning_rate * delta
	return final_error

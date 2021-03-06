import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from os import system


class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes


        self.lr = learningrate 
        

        self.wih = np.random.normal(0.0, self.hnodes**(-0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, self.onodes**(-0.5), (self.onodes, self.hnodes))


        self.activation_function =lambda x: scipy.special.expit(x)
        

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T
        
        #Hidden IO signals
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #Outputs IO signals
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)


        output_errors = targets - final_outputs

        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot(output_errors * final_outputs * (1.0 - final_outputs), hidden_outputs.T)

        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), inputs.T)


    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin = 2).T

        #hidden IO signals
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        #outputs IO signals
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

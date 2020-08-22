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


def main():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    learning_rate = 0.1

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    #Learning
    training_data_file = open('mnist_dataset/mnist_train_100.csv', 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = 2 

    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
    

    #Testing
    test_data_file = open('mnist_dataset/mnist_test_10.csv', 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
  
    scored = []

    for record in test_data_list:

        all_values = record.split(',')

        correct_label = int(all_values[0])

        inputs = np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.1

        outputs = n.query(inputs)

        label = np.argmax(outputs)

        print('True marker: ', correct_label)
        print('Network marker:', label)

        if label == correct_label:
            scored.append(1)
        else:
            scored.append(0)


    print(scored)
    scorecard = np.asarray(scored)
    print('Efficient: ', scorecard.sum()/scorecard.size)


'''
    test_number = int(input('Enter the test number 0-9 : '))
    all_values = test_data_list[test_number].split(',')

    result = n.query(np.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01)

    print(all_values[0])
    print(result)
'''

if __name__ == '__main__':
    main()

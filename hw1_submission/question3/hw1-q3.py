#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def hot_encode(position, size):
    
    encoded = np.zeros(size)
    encoded[position] = 1

    return encoded

def hot_decode(vector):
    
    decoded = np.zeros(len(vector))
    
    for i in range(0, len(vector)):
        
        decoded[i] = np.argmax(vector[i])

    return decoded

def softmax(vector):
#Define Softmax function   
    vector -= vector.max() #remove highest value to make the softmax function numerically stable
    e = np.exp(vector)
    
    return e / e.sum()
    


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        
        #Get Prediction
        y_i_hat = self.predict(x_i)
        
        #Detect if the predicted class is wrong
        if (y_i_hat - y_i) != 0:
                
            self.W[y_i_hat] = self.W[y_i_hat] - x_i  #Decrease weight of incorrect predicted class
            self.W[y_i] = self.W[y_i] + x_i  #Increase weight of correct class
        


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q3.1b
        
        #Prediction
        z = np.dot(self.W, x_i)
        gradient = np.outer(softmax(z), x_i)

        #Update Weights for correct prediciton
        self.W[y_i] = self.W[y_i] + learning_rate*x_i  
        
        #Update Weights for all classes
        self.W = self.W - learning_rate*gradient
        


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, layers):
        # Initialize an MLP with a single hidden layer.
        
        #Define initial weights and biases
        self.units = [n_features, hidden_size, n_classes]
        W1 = np.random.normal(loc=0.1,scale=0.1,size=(self.units[1], self.units[0]))
        b1 = np.zeros(self.units[1])
        W2 = np.random.normal(loc=0.1,scale=0.1,size=(self.units[2], self.units[1]))
        b2 = np.zeros(self.units[2])
        
        self.W = [W1, W2]
        self.B= [b1, b2]
        
        #Define the Relu Function and its derivative
        self.relu = lambda x : np.maximum(0, x)
        self.relu_derivative = lambda h : np.greater(h, 0).astype(int)
        
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i, y_i: a single training example
        This function makes an update to the model weights
        """
        num_layers = len(self.W)
        grad_weights, grad_biases = self.backward(x_i, y_i)
        for i in range(num_layers):
            self.W[i] -= learning_rate*grad_weights[i]
            self.B[i] -= learning_rate*grad_biases[i]
        

    def predict(self, X):
        """
        X (n_examples x n_features)
        Compute the forward pass of the network. 
        """
        
        predicted = []
        #Compute each example in the dataset
        for x in X:
            output, _ = self.forward(x)
            y_hat = softmax(output)
            predicted.append(y_hat)
            
        predicted = np.array(predicted)
        return predicted
        

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        
        #Decode to normal encoding in order to compare results
        y_hat_dec = hot_decode(y_hat)
        
        n_correct = (y == y_hat_dec).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i)
            
            
    def forward(self, X):
        """
        X (n_examples x n_features)
        Compute the forward pass of the network, to this function 
        it's only passed one example at a time
        """
        num_layers = len(self.W)
        g = self.relu
        hiddens = []
        for i in range(0, num_layers):
            h = X if i == 0 else hiddens[i-1]
            z = self.W[i].dot(h) + self.B[i]
            
            if i < num_layers-1:  # Assume the output layer has no activation.
                hiddens.append(g(z))
                
        output = z
        # For classification this is a vector of logits (label scores).
        # For regression this is a vector of predictions.
        return output, hiddens
    
    def backward(self, x_i, y_i):
        """
        x_i, y_i: a single training example
        Compute the backward pass of the network
        """
        
        num_layers = len(self.W)
        g = self.relu
        z, hiddens = self.forward(x_i)
        
        grad_weights = []
        grad_biases = []
        
        enc_y_i = hot_encode(y_i, 10) #10 is the number of classes
        grad_z = softmax(z) - enc_y_i  # Grad of loss wrt last z.
        
        for i in range(num_layers-1, -1, -1):
            
            # Gradient of hidden parameters.
            h = x_i if i == 0 else hiddens[i-1]
            grad_weights.append(grad_z[:, None].dot(h[:, None].T))
            grad_biases.append(grad_z)
            
            # Gradient of hidden layer below.
            grad_h = self.W[i].T.dot(grad_z)
            
            # Gradient of hidden layer below before activation.
            assert(g == self.relu)
            grad_z = grad_h * self.relu_derivative(h)# Grad of loss wrt z3. only valid if relu function used
            #For the ReLU function derivating the z or the h for the same layer l will give the same result
            
        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases

def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")  #default=20
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()
    
    utils.configure_seed(seed=42)
    
    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    
    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]
    
    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))
    
    # plot
    plot(epochs, valid_accs, test_accs)

if __name__ == '__main__':
    main()

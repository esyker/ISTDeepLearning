#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
import numpy as np

import utils

class CNN(nn.Module):
    
    def __init__(self):
        """
        The __init__ should be used to declare what kind of layers and other
        parameters the module has. For example, a CNN module has convolution,
        max pooling, activation, linear, and other types of layers. For an 
        idea of how to us pytorch for this have a look at
        https://pytorch.org/docs/stable/nn.html
        """
        """
        • A first block comprising (order as provided):
        – A convolution layer with 16 output channels, a kernel of size 3x3, stride of 1, and
        padding chosen to preserve the original image size.
        – A rectified linear unit activation function.
        – A max pooling with kernel size 2x2 and stride of 2.
        
        
        • A second block comprising (order as provided):
        – A convolution layer with 32 output channels, a kernel of size 3x3, stride of 1, and
        padding of zero.
        – A rectified linear unit activation function.
        – A max pooling with kernel size 2x2 and stride of 2.
        • An affine transformation with 600 output features (to determine the number of input
        features use the number of channels, width and height of the output of the second block.
        Hint: The number of input features = number of output channels × output width ×
        output height).
        • A dropout layer with a dropout probability of your choice.
        • An affine transformation with 120 output features.
        • An affine transformation with the number of classes followed by an output LogSoftmax
        layer.
        Hint: use the functions nn.Sequential, nn.Conv2d and nn.MaxPool2d.
        """
        super(CNN,self).__init__()

        self.convblock1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        #input size 28 (28X28 images)
        #padding to maintain size = (kernel_size-1)/2 = (3-1)/2=1
        #Convolution size
        #output size formula = ((W-K+2*P)/S)+1   W:input_size K:kernel_size P:padding S:stride
        #output size (28 -3 + 2*1)/1 +1 = 28
        #Max pool size
        #output size = 28/2 = 14
        
        self.convblock2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        #output size (14 -3 + 2*0)/1 +1 = 12
        #output size = 12/2 = 6
        
        self.linear_layers = nn.Sequential(nn.Flatten(),
                                           nn.Linear(1152, 600),nn.ReLU(),nn.Dropout(0.3),
                                           nn.Linear(600, 120),nn.ReLU(),
                                           nn.Linear(120, 10),nn.LogSoftmax())

        self.feedforward = nn.Sequential(self.convblock1,self.convblock2,self.linear_layers)
        
    def forward(self, x):
        """
        x (batch_size x n_channels x height x width): a batch of training 
        examples

        Every subclass of nn.Module needs to have a forward() method. forward()
        describes how the module computes the forward pass. This method needs 
        to perform all the computation needed to compute the output from x. 
        This will include using various hidden layers, pointwise nonlinear 
        functions, and dropout. Don't forget to use logsoftmax function before 
        the return

        One nice thing about pytorch is that you only need to define the
        forward pass -- this is enough for it to figure out how to do the
        backward pass.
        """
        return self.feedforward(x)
        

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function

    To train a batch, the model needs to predict outputs for X, compute the
    loss between these predictions and the "gold" labels y using the criterion,
    and compute the gradient of the loss with respect to the model parameters.

    Check out https://pytorch.org/docs/stable/optim.html for examples of how
    to use an optimizer object to update the parameters.

    This function should return the loss (tip: call loss.item()) to get the
    loss as a numerical value that is not part of the computation graph.
    """
    optimizer.zero_grad()
    yhat = model(X, **kwargs)
    loss = criterion(yhat, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def plot_kernels(convolutional_layer_1, convolutional_layer_2, name_1='', name_2=''):
    kernels_l1 = convolutional_layer_1.weight.detach().clone()

    # normalize to (0,1) for better visualization
    kernels_l1 = kernels_l1 - kernels_l1.min()
    kernels_l1 = kernels_l1 / kernels_l1.max()
    filter_img_l1 = torchvision.utils.make_grid(kernels_l1, nrow = 8)
    # change ordering as matplotlib expects data as (H, W, C)
    plt.clf()
    plt.imshow(filter_img_l1.permute(1, 2, 0))
    plt.savefig('%s.pdf' % (name_1), bbox_inches='tight')

    kernels_l2 = convolutional_layer_2.weight.detach().clone()

    # normalize to (0,1) for better visualization
    kernels_l2 = kernels_l2 - kernels_l2.min()
    kernels_l2 = kernels_l2 / kernels_l2.max()
    filter_img_l2 = torchvision.utils.make_grid(torch.from_numpy(np.expand_dims(kernels_l2[:,0,:,:],1)), nrow = 8)
    # change ordering as matplotlib expects data as (H, W, C)
    plt.clf()
    plt.imshow(filter_img_l2.permute(1, 2, 0))
    plt.savefig('%s.pdf' % (name_2), bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=15, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.3)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-convlayer1', help="""Variable name holding the 1st convolutional layer.""")
    parser.add_argument('-convlayer2', help="""Variable name holding the 2nd convolutional layer.""")
    
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_classification_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    # initialize the model
    model = CNN()
    
    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )
    
    # get a loss criterion
    criterion = nn.NLLLoss()
    
    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    config = "{}-{}-{}-{}".format(opt.learning_rate, opt.dropout, opt.l2_decay, opt.optimizer)

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
    
    if (opt.convlayer1 and opt.convlayer2) is not None:
        conv_layer_1_viz = eval('model.' + opt.convlayer1)
        conv_layer_2_viz = eval('model.' + opt.convlayer2)
        plot_kernels(conv_layer_1_viz, conv_layer_2_viz, name_1='CNN-conv_layer_1_filters-{}'.format(config), name_2='CNN-conv_layer_2_filters-{}'.format(config))

if __name__ == '__main__':
    main()

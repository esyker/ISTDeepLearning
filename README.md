# Deep Learning projects

## Project 2 (hw2)

Check hw2/report2.pdf to see the developed work.

**Image classification** with CNNs. In this exercise, created a convolutional neural
network to perform classification using the Fashion-MNIST dataset.

**Image captioning** is the problem of automatically describing an image
with a short text description. This task is usually addressed with an encoder-decoder model.
Typically, the encoder is a CNN model that processes and summarises the input image into
relevant feature representations. The representations are then passed to a language decoder,
commonly an LSTM model, which generates the respective caption word-by-word given the
previous words and the image encoder representations.

Added an attention mechanism to the decoder (additive attention), which
weights the contribution of the different image regions, according to relevance for the
current prediction.

## Project 1 (hw1)

Check hw1/report1.pdf to see the developed work.

**Image classification** with an autodiff toolkit. In the previous question, you had to write
gradient backpropagation by hand. This time, you will implement the same system using a
deep learning framework with automatic differentiation. Pytorch skeleton code is provided
(hw1-q4.py) but if you feel more comfortable with a different framework, you are free to use it
instead.

**Image classification** with linear classifiers and neural networks. In this exercise, you
will implement a linear classifier for a simple image classification problem, using the FashionMNIST dataset.



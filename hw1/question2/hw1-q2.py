import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


def distance(analytic_solution, model_params):
    return np.linalg.norm(analytic_solution - model_params)


def solve_analytically(X, y):
    """
    X (n_points x n_features)
    y (vector of size n_points)

    Q2.1: given X and y, compute the exact, analytic linear regression solution.
    This function should return a vector of size n_features (the same size as
    the weight vector in the LinearRegression class).
    """
    #If A is full rank, x=AT(AAT)−1b is an exact solution where Ax=b and it also has minimal ∥x∥.
    #https://math.stackexchange.com/questions/1745101/least-squares-with-singular-aat
    """u, sigma, vt = np.linalg.svd(X)
    sigma_plus = (sigma.T.dot(sigma)).dot(sigma.T)
    svd_pseudo_inverse= vt.T.dot(sigma_plus).dot(u.T)
    return svd_pseudo_inverse.dot(y)
    """    
    return np.linalg.pinv(X).dot(y)
    
    


class _RegressionModel:
    """
    Base class that allows evaluation code to be shared between the
    LinearRegression and NeuralRegression classes. You should not need to alter
    this class!
    """
    def train_epoch(self, X, y, **kwargs):
        """
        Iterate over (x, y) pairs, compute the weight update for each of them.
        Keyword arguments are passed to update_weight
        """
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def evaluate(self, X, y):
        """
        return the mean squared error between the model's predictions for X
        and he ground truth y values
        """
        yhat = self.predict(X)
        error = yhat - y
        squared_error = np.dot(error, error)
        mean_squared_error = squared_error / y.shape[0]
        return np.sqrt(mean_squared_error)


class LinearRegression(_RegressionModel):
    def __init__(self, n_features, **kwargs):
        self.w = np.zeros((n_features))

    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        Q2.2a

        x_i, y_i: a single training example

        This function makes an update to the model weights (in other words,
        self.w).
        """
        y_i_hat = self.predict(x_i)
        error = y_i_hat - y_i
        gradient = x_i.T.dot(error)
        self.w-=learning_rate*gradient

    def predict(self, X):
        return np.dot(X, self.w)


class NeuralRegression(_RegressionModel):
    """
    Q2.2b
    """
    def __init__(self, n_features, hidden):
        """
        In this __init__, you should define the weights of the neural
        regression model (for example, there will probably be one weight
        matrix per layer of the model).
        """
        self.units = [n_features, hidden, 1]
        W1 = np.random.normal(loc=0.1,scale=0.1,size=(self.units[1], self.units[0]))
        b1 = np.zeros(self.units[1])
        W2 = np.random.normal(loc=0.1,scale=0.1,size=(self.units[2], self.units[1]))
        b2 = np.zeros(self.units[2])
        self.W = [W1, W2]
        self.B= [b1,b2]
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
        X: a (n_points x n_feats) matrix.

        This function runs the forward pass of the model, returning yhat, a
        vector of size n_points that contains the predicted values for X.

        This function will be called by evaluate(), which is called at the end
        of each epoch in the training loop. It should not be used by
        update_weight because it returns only the final output of the network,
        not any of the intermediate values needed to do backpropagation.
        """
        predicted = []
        for x in X:
            output, _ = self.forward(x)
            y_hat = output[0]
            predicted.append(y_hat)
        predicted = np.array(predicted)
        return predicted
    
    def forward(self, X):
        num_layers = len(self.W)
        g = self.relu
        hiddens = []
        for i in range(num_layers):
            h = X if i == 0 else hiddens[i-1]
            z = self.W[i].dot(h) + self.B[i]
            if i < num_layers-1:  # Assume the output layer has no activation.
                hiddens.append(g(z))
        output = z
        # For classification this is a vector of logits (label scores).
        # For regression this is a vector of predictions.
        return output , hiddens
    
    def backward(self, x_i, y_i):
        num_layers = len(self.W)
        g = self.relu
        z, hiddens = self.forward(x_i)
        grad_z = z - y_i  # Grad of loss wrt last z.
        grad_weights = []
        grad_biases = []
        for i in range(num_layers-1, -1, -1):
            # Gradient of hidden parameters.
            h = x_i if i == 0 else hiddens[i-1]
            grad_weights.append(grad_z[:, None].dot(h[:, None].T))
            grad_biases.append(grad_z)
            # Gradient of hidden layer below.
            grad_h = self.W[i].T.dot(grad_z)
            # Gradient of hidden layer below before activation.
            assert(g == self.relu)
            grad_z = grad_h *self.relu_derivative(z)# Grad of loss wrt z3. only valid if relu function used
        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases


def plot(epochs, train_loss, test_loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, epochs[-1] + 1, step=10))
    plt.plot(epochs, train_loss, label='train')
    plt.plot(epochs, test_loss, label='test')
    plt.legend()
    plt.show()


def plot_dist_from_analytic(epochs, dist):
    plt.xlabel('Epoch')
    plt.ylabel('Dist')
    plt.xticks(np.arange(0, epochs[-1] + 1, step=10))
    plt.plot(epochs, dist, label='dist')
    plt.legend()
    plt.show()


#def main():
parser = argparse.ArgumentParser()
parser.add_argument('model', choices=['linear_regression', 'nn'],
                    help="Which model should the script run?")
parser.add_argument('-epochs', default=150, type=int,
                    help="""Number of epochs to train for. You should not
                    need to change this value for your plots.""")
parser.add_argument('-hidden_size', type=int, default=150)
parser.add_argument('-learning_rate', type=float, default=0.001)
opt = parser.parse_args()

utils.configure_seed(seed=42)

add_bias = opt.model != 'nn'
data = utils.load_regression_data(bias=add_bias)
train_X, train_y = data["train"]
test_X, test_y = data["test"]

n_points, n_feats = train_X.shape

# Linear regression has an exact, analytic solution. Implement it in
# the solve_analytically function defined above.
if opt.model == "linear_regression":
    analytic_solution = solve_analytically(train_X, train_y)
else:
    analytic_solution = None

# initialize the model
if opt.model == "linear_regression":
    model = LinearRegression(n_feats)
else:
    model = NeuralRegression(n_feats, opt.hidden_size)

# training loop
epochs = np.arange(1, opt.epochs + 1)
train_losses = []
test_losses = []
dist_opt = []
for epoch in epochs:
    print('Epoch %i... ' % epoch)
    train_order = np.random.permutation(train_X.shape[0])
    train_X = train_X[train_order]
    train_y = train_y[train_order]
    model.train_epoch(train_X, train_y, learning_rate=opt.learning_rate)

    # Evaluate on the train and test data.
    train_losses.append(model.evaluate(train_X, train_y))
    test_losses.append(model.evaluate(test_X, test_y))

    if analytic_solution is not None:
        model_params = model.w
        dist_opt.append(distance(analytic_solution, model_params))

    print('Loss (train): %.3f | Loss (test): %.3f' % (train_losses[-1], test_losses[-1]))

plot(epochs, train_losses, test_losses)
if analytic_solution is not None:
    plot_dist_from_analytic(epochs, dist_opt)


#if __name__ == '__main__':
#    main()

#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import time
import utils

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
        X (n_examples x n_features)
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
        # Q1.1a

        # Calculate predicted y
        y_hat = self.predict(x_i)
        # eta = kwargs.get("learning_rate", 1)

        # update weights if prediction is incorrect
        if y_hat != y_i: 
            self.W[y_i] += y_i * x_i  
            self.W[y_hat] -= y_i * x_i

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """

        # Compute scores for the classes:
        scores = np.dot(self.W, x_i)

        # Softmax transformation
        exp_scores = np.exp(scores)        
        probs = exp_scores / np.sum(exp_scores)

        # Compute the gradient of the loss with respect to the weights
        one_hot_y = np.zeros_like(scores)
        one_hot_y[y_i] = 1
        gradient = np.outer(probs - one_hot_y, x_i) + l2_penalty * self.W

        # Update the weights using the gradient
        self.W -= learning_rate * gradient

class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        
        # Input-to-hidden layer
        self.W1 = np.random.normal(0.1, 0.1, (n_features, hidden_size))
        self.b1 = np.zeros((1, hidden_size))

        # Hidden-to-output layer
        self.W2 = np.random.normal(0.1, 0.1, (hidden_size, n_classes))
        self.b2 = np.zeros((1, n_classes))

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Compute the derivative of ReLU."""
        return np.where( x > 0, 1, 0)

    def softmax(self, x):
        x = x - np.max(x)
        probs = np.exp(x) / np.sum(np.exp(x))
        return probs

    def predict(self, X):
        hidden = self.relu(np.dot(X, self.W1) + self.b1)
        return np.argmax(np.dot(hidden, self.W2) + self.b2, axis=1)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible
    
    def forward_pass(self, X):
        """Compute the forward pass."""
        hidden = self.relu(np.dot(X, self.W1) + self.b1)
        logits = np.dot(hidden, self.W2) + self.b2
        output = self.softmax(logits)
        return hidden, output
    
    def compute_loss(self, output, target):
        """Compute the cross-entropy loss."""
        one_hot_target = np.zeros_like(output)
        one_hot_target[np.arange(len(target)), target] = 1
        return -np.sum(one_hot_target * np.log(output + 1e-6))  # Add epsilon for numerical stability
    
    def backpropagate(self, X, hidden, output, target):
        """Compute the gradients for weights and biases."""
        one_hot_target = np.zeros_like(output)
        one_hot_target[np.arange(len(target)), target] = 1
        d_output = output - one_hot_target
        
        grad_W2 = np.dot(hidden.T, d_output)
        grad_b2 = d_output

        d_hidden = np.dot(d_output, self.W2.T) * self.relu_derivative(hidden)
        grad_W1 = np.dot(X.T, d_hidden)
        grad_b1 = d_hidden
        
        return grad_W1, grad_b1, grad_W2, grad_b2

    def train_epoch(self, X, y, learning_rate=0.001, **kwargs):
        """
        Train for one epoch using stochastic gradient descent (no batching).
        Returns the average loss for the epoch.
        """
        total_loss = 0
        num_samples = X.shape[0]

        # Iterate over each sample
        for x_i, y_i in zip(X, y):
            # Get the current sample
            y_i = np.array([y_i])
            # Forward pass
            hidden, output = self.forward_pass(x_i.reshape(1, -1))

            # Compute loss (cross-entropy)
            loss = self.compute_loss(output, y_i)
            total_loss += loss

            # Backpropagation
            grad_W1, grad_b1, grad_W2, grad_b2 = self.backpropagate(x_i.reshape(1, -1), hidden, output, y_i)

            # Update weights and biases
            self.W1 -= learning_rate * grad_W1
            self.b1 -= learning_rate * grad_b1
            self.W2 -= learning_rate * grad_W2
            self.b2 -= learning_rate * grad_b2

        # Return the average loss for the epoch
        return total_loss / num_samples

def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_w_norm(epochs, w_norms, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")

if __name__ == '__main__':
    main()

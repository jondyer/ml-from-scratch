"""
Intro to Machine Learning
@author: Jonathan Dyer (jbdyer)
"""
import math
import csv
import sys
import numpy as np
from collections import Counter

############################## Global Variables ################################
_vocab = {}
LEARNING_RATE = 0.1

############################## Helper Math #####################################
def sigmoid(z):
    return (1.0 / (1 + np.exp(-z)))

def sparse_dot(X, Y):
    """
    Takes a 'sparse array' X (in dict form as below) and computes the dot
    product with the normal list-like array Y.

    Parameters
    ----------
    X : dict
        Of the form {idx : value} where idx is the index in an equivalent
        dense array. Only nonzero elements are contained here.
    Y : list-like
        The vector to dot with X, contains real values.

    Returns
    -------
    float
        The scalar dot product of the two vectors.
    """
    product = 0.0
    for i, x in X.items():
        product += x * Y[i]
    return product

def sparse_add(X, Y):
    """Expecting X to be the sparse one."""
    Y_new = Y.copy()
    for i, x in X.items():
        Y_new[i] += x
    return Y_new

def sparse_sub(X, Y):
    """Expecting X to be the sparse one."""
    Y_new = Y.copy()
    for i, x in X.items():
        Y_new[i] -= x
    return Y_new


def compute_error(X, Y):
    return np.mean(np.array(X) != np.array(Y))


############################## Learning Functions ##############################
def SGD(theta, eta, x_i, y_i):
    # calc the scalar constant for this data point
    q = y_i - sigmoid(sparse_dot(x_i, theta))

    # now multiply the two constants
    c = q * eta

    # and multiply every entry of x_i by c in preparation for the update
    x_i = {i: val*c for i, val in x_i.items()}

    # finally perform the update
    for i, x in x_i.items():
        theta[i] += x

    return theta


def predict(X, theta):
    Y = []
    for x_i in X:
        Y.append(predict_single(x_i, theta))
    return Y


def predict_single(x_i, theta):
    prob = sparse_dot(x_i, theta)
    if prob > 0:
        return 1
    else:
        return 0


def neg_log_likelihood(theta, X, y):
    total = 0.0
    for idx, x_i in enumerate(X):
        t_x = sparse_dot(x_i, theta)
        term_1 = -y[idx] * t_x
        term_2 = np.log(1 + np.exp(t_x))
        total += term_1 + term_2

    return total

############################## File I/O ########################################
def read_vocab(filename):
    with open(filename, 'r') as file:
        lines = map(lambda x : x.split(), file.readlines())
        return {row[0] : int(row[1]) for row in lines}


def read_formatted_data(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter='\t')
        lines = list(reader)

    # first retrieve labels
    y = [int(line.pop(0)) for line in lines]

    # store as list of dicts, making sure to include bias feature!
    # to account for the bias now being '0', must increment the keys
    X = []
    for line in lines:
        x = dict([[int(elem.split(':')[0])+1, int(elem.split(':')[1])] for elem in line])
        x[0] = 1
        X.append(x)

    return y, X


############################## Main method ###################################
def main():
    train_in = sys.argv[1]
    valid_in = sys.argv[2]
    test_in = sys.argv[3]
    dict_in = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epochs = int(sys.argv[8])


    ########################### PART 1: Read in formatted data #################
    # 1a: Read in the dictionary
    global _vocab
    _vocab = read_vocab(dict_in)

    # 1b: Read in the training, validation, and test data (sparsely represented)
    train_labels, train_data = read_formatted_data(train_in)
    valid_labels, valid_data = read_formatted_data(valid_in)
    test_labels, test_data = read_formatted_data(test_in)


    ########################### PART 2: Train learner ##########################
    # 2a: Set up variables etc.
    # theta will be of length M+1 where M is the size of the vocab
    M = len(_vocab)
    theta = np.zeros(M+1)       # initialize all params to 0 (folding in bias)

    # 2b: Perform SGD to minimize the objective fn
    # for num_epochs, iterate over the entire dataset
    for epoch in range(num_epochs):     # for specified iterations
        print("Epoch:", epoch)    # debug
        for i, x_i in enumerate(train_data):                # for each data point
            SGD(theta, LEARNING_RATE, x_i, train_labels[i])

    ########################### PART 3: Test learner ###########################
    # 3a: Get predictions for all labels
    pred_train_labels = predict(train_data, theta)
    pred_test_labels = predict(test_data, theta)

    # 3b: Calculate errors
    err_train = compute_error(train_labels, pred_train_labels)
    err_test = compute_error(test_labels, pred_test_labels)

    ########################### PART 4: Write out data #########################
    # 4a: Write out predicted labels
    with open(train_out, 'w') as tn_label_file:
        for l in pred_train_labels:
            tn_label_file.write(str(l) + '\n')

    with open(test_out, 'w') as tt_label_file:
        for l in pred_test_labels:
            tt_label_file.write(str(l) + '\n')

    # 4b: Write out metrics
    with open(metrics_out, 'w') as met_file:
        met_file.write('error(train): ' + str(err_train) + '\n')
        met_file.write('error(test): ' + str(err_test) + '\n')



if __name__ == '__main__':
    if len(sys.argv) == 9:
        main()
    else:
        print("Wrong number of arguments!\n"
              + "Correct usage is 'python feature.py <args... (8 args)>'")

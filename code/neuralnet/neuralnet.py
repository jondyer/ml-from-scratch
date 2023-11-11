"""
Intro to Machine Learning
@author: Jonathan Dyer (jbdyer)
"""
import math
import csv
import sys
import numpy as np


############################## Global Variables ################################
_learning_rate = 0.0

############################## Neural Net Modules ##############################
####### Linear Module #######
def linearForward(v, weights):
    r = v @ np.transpose(weights)
    return r

def linearBackward(x, weights, g):
    g_weights = np.outer(g, x)
    g_x = g @ weights
    return g_weights, g_x


####### Sigmoid Module #######
def sigmoidForward(z):
    """
    Returns element-wise sigmoid of input (must be scalar or numpy array).
    """
    return (1.0 / (1 + np.exp(-z)))

def sigmoidBackward(z, g_z):
    dz_da = z - (z * z)
    g_a = g_z * dz_da       # element-wise
    return g_a


####### Softmax and CrossEntropy Module #######
def softmaxForward(v):
    numerators = np.exp(v)      # elem-wise
    total = np.sum(numerators)
    r = numerators / total
    return r

def crossEntropyForward(y, y_hat):
    log_y_hat = np.log(y_hat)
    s = np.dot(log_y_hat, y)
    return -s

def crossMaxBackward(y_i, y_hat):
    return y_hat - y_i


############################## Helper Functions ################################
####### one forward pass to find mean crossentropy #######
def meanCrossEntropy(X, Y, alpha, beta):
    total_loss = 0.0

    for i, x_i in enumerate(X):                # for each data point
        y_i = Y[i]

        # NNForward: get o = (a, z, b, ̂y, and J) — all vectors except loss (scalar)
        o = forwardNN(x_i, y_i, alpha, beta)

        # add the loss for this data point
        total_loss += o[4]

    return total_loss / len(X)


def compute_error(X, Y):
    return np.mean(np.array(X) != np.array(Y))

def one_hot(Y_raw):
    Y = []
    for y in Y_raw:
        y_hot = np.zeros(10)
        y_hot[y] = 1.0
        Y.append(y_hot)

    return np.array(Y)

############################## Learning Functions ##############################
def forwardNN(x_i, y_i, alpha, beta):
    a = linearForward(x_i, alpha)
    z_star = sigmoidForward(a)              # this z has no bias term
    z = np.insert(z_star, 0, 1, axis=1)     # this z has a bias term
    b = linearForward(z, beta)
    y_hat = softmaxForward(b)
    J = crossEntropyForward(y_i, y_hat)
    o = (a, z, b, y_hat, J)

    return o


def backwardNN(x_i, y_i, alpha, beta, o):
    # get intermediate values -- note z has a bias element already
    a, z, b, y_hat, J = o
    beta_star = beta[:, 1:]
    z_star = z[0,1:]

    # start calculating gradients (skip g_y_hat)
    g_b = crossMaxBackward(y_i, y_hat)
    g_beta, g_z = linearBackward(z, beta_star, g_b)
    g_a = sigmoidBackward(z_star, g_z)
    g_alpha, _ = linearBackward(x_i, alpha, g_a)    # discard g_x

    return g_alpha, g_beta


def SGD(alpha, beta, train_X, train_Y):
    """
    Performs one unshuffled SGD pass for the given data set.
    """
    global _learning_rate

    for i, x_i in enumerate(train_X):                # for each data point
        y_i = train_Y[i]

        # NNForward: get o = (a, z, b, ̂y, and J) — all vectors except loss (scalar)
        o = forwardNN(x_i, y_i, alpha, beta)

        # NNBackward: use o to get ∇α and ∇β — same size as α and β
        g_alpha, g_beta = backwardNN(x_i, y_i, alpha, beta, o)

        # Update weight matrices
        alpha -= (_learning_rate * g_alpha)
        beta -= (_learning_rate * g_beta)

    return alpha, beta


def predict(X, alpha, beta):
    Y = []
    for x_i in X:
        a = linearForward(x_i, alpha)
        z_star = sigmoidForward(a)              # this z has no bias term
        z = np.insert(z_star, 0, 1, axis=1)     # this z has a bias term
        b = linearForward(z, beta)
        y_hat = softmaxForward(b)

        # finally get label as argmax
        l = np.argmax(y_hat)
        Y.append(l)
    return Y


############################## Weight Init #####################################
def random_init(alpha_dim, beta_dim):
    # use numpy random to generate the appropriate matrix
    alpha = np.random.uniform(-0.1, 0.1, alpha_dim)
    beta = np.random.uniform(-0.1, 0.1, beta_dim)

    # set bias columns to 0
    alpha[:, 0] = 0
    beta[:, 0] = 0

    return alpha, beta


def zero_init(alpha_dim, beta_dim):
    # initialize the matrices
    alpha = np.zeros(alpha_dim, dtype=float)
    beta = np.zeros(beta_dim, dtype=float)

    return alpha, beta

############################## File I/O ########################################
def read_data(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        lines = list(reader)

    # first retrieve labels
    Y = [int(line[0]) for line in lines]

    # now initialize bias term for each data point (i.e. x_0 = 1)
    X = []
    for line in lines:
        line[0] = 1
        X.append(np.array([line], dtype=float))

    X = np.array(X)

    return Y, X

############################## Main method ###################################
def main():
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epochs = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])

    ########################### PART 0: A few assertions... ####################
    assert init_flag==1 or init_flag==2, "Wrong init_flag argument, pick 1 or 2."
    assert learning_rate > 0, "Learning rate must be positive."
    assert hidden_units > 0, "Number of hidden units must be positive."
    assert num_epochs > 0, "Number of epochs must be positive."


    ########################### PART 1: Read in formatted data #################
    # 1a: Read in the training and test data from csv files as numpy arrays
    train_labels_raw, train_data = read_data(train_in)
    test_labels_raw, test_data = read_data(test_in)

    train_labels = one_hot(train_labels_raw)
    test_labels = one_hot(test_labels_raw)

    # 1b: Set up global variables, constants, etc.
    global _learning_rate
    _learning_rate = learning_rate


    ########################### PART 2: Train learner ##########################
    # 2a: Initialize weight matrices (α and β)
    #   get dims
    D = hidden_units
    M = train_data.shape[2] - 1     # account for bias column
    K = 10                          # constant for 10 output classes

    #   check flag and init
    if init_flag==1:    # RANDOM init
        alpha, beta = random_init((D, M+1), (K, D+1))
    else:               # ZERO init
        alpha, beta = zero_init((D, M+1), (K, D+1))


    # 2b: Perform SGD to minimize the objective fn (cross-entropy loss)
    mets = []       # to hold metrics as we go

    # for num_epochs, iterate over the entire dataset in an SGD pass
    # alpha and beta are modified in-place
    for epoch in range(num_epochs):
        learned_alpha, learned_beta = SGD(alpha, beta, train_data, train_labels)

        # Compute mean cross-entropy -- go FORWARD through both datasets with new
        #   and improved α and β, and calculate mean loss
        mean_loss_train = meanCrossEntropy(train_data, train_labels, alpha, beta)
        mean_loss_test = meanCrossEntropy(test_data, test_labels, alpha, beta)

        mets.append("Epoch=" + str(epoch+1) + " crossentropy(train): " + str(mean_loss_train))
        mets.append("Epoch=" + str(epoch+1) + " crossentropy(test): " + str(mean_loss_test))

    # debug
    # for line in mets:
    #     print(line)

    ########################### PART 3: Test learner ###########################
    # 3a: Get predictions for all labels
    pred_train_labels = predict(train_data, alpha, beta)
    pred_test_labels = predict(test_data, alpha, beta)

    # 3b: Calculate errors
    err_train = compute_error(train_labels_raw, pred_train_labels)
    err_test = compute_error(test_labels_raw, pred_test_labels)

    mets.append("error(train): " + str(err_train))
    mets.append("error(test): " + str(err_test))

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
        for item in mets:
            met_file.write(item + '\n')



if __name__ == '__main__':
    if len(sys.argv) == 10:
        main()
    else:
        print("Wrong number of arguments!\n"
              + "Correct usage is 'python neuralnet.py <args... (9 args)>'")

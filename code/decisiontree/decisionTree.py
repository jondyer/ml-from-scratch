# -*- coding: utf-8 -*-
"""
Intro to Machine Learning
@author: Jonathan Dyer (jbdyer)
"""
import math
import csv
import sys
import numpy as np

################ Global Variables ################
MAX_DEPTH = 0
_attributes = {}
_train_labels = []
_test_labels = []
_unique_labels = []

################ Node Class ################
class Node:
    def __init__(self, att_name, directions, left, right):
        self.att_name = att_name        # assign the splitting attribute
        self.directions = directions    # this helps test points navigate
        self.left = left                # left child (Node or Leaf)
        self.right = right              # right child (Node or Leaf)

    def get_idx(self):
        return _attributes[self.att_name]

class Leaf(Node):
    def __init__(self, prediction):
        self.prediction = prediction    # this is the predicted label

################ Helper Functions ################
def prob(val, data):
    """
    Calculates the simple proportion (probability) of val in data.

    Parameters
    ----------
    data : numpy array
        Expected to be a vector of attribute values
    val : anything
        A possible attribute in data
    """
    return list(data).count(val) / len(data)


def entropy(Y):
    """
    Basic entropy calculation. Expects a vector.
    """
    total = 0
    for i in set(Y):             # for each distinct value of Y
        p = prob(i, Y)           # get probability of Y taking that value
        if p > 0:
            log_p = np.log2(p)
        else:
            log_p = 0
        total += (p * log_p)

    return -1 * total


def joint_entropy(X,Y):
    """
    Calculates the joint entropy of two different vectors of data.
    """
    list_of_probs = []          # To store all the probabilities we get.
    X = np.array(X)             # Need to make sure we have numpy arrays
    Y = np.array(Y)             # (hopefully they're the same length!).

    for x_val in set(X):        # For each distinct value of X...
        for y_val in set(Y):    # for each distinct value of Y...
            # get an array of booleans that tell us where they co-occur
            intersection = np.array((X == x_val) & (Y == y_val))

            # then count the true times
            co_occur = np.sum(intersection)
            joint_prob = co_occur / len(X)  # divide to get probability

            list_of_probs.append(joint_prob)    # add to the list

    # turn it into a numpy array for quicker math (and remove 0 values!)
    list_of_probs = np.array([i for i in list_of_probs if i > 0])

    # then multiply by log2(prob) --> make sure m isn't 0!
    updated_probs = np.array([m * np.log2(m) for m in list_of_probs])

    # finally sum them all up and multiply by -1
    return np.sum(updated_probs) * -1


def mutual_info(Y, X):
    """
    Returns the mutual information of these two columns of values.
    """
    # print("Enter mutual info")
    return entropy(Y) + entropy(X) - joint_entropy(X,Y)


def get_majority(Y):
    """
    Assumes that the labels are in the last column of the data, or that
    the data is simply a vector of labels.

    Returns the label that takes the majority vote in the given data.
    """
    # check if this is a vector...
    if len(np.shape(Y)) < 2:     # we have a vector of only labels
        labels = np.array(Y)
    else:                           # otherwise get the vector of labels
        labels = np.array(Y)[:,-1]

    # get counts of each label
    unique_labels, counts = np.unique(labels, return_counts=True)
    # print(unique_labels, counts)

    # now the index of the moximum frequency label
    i = np.argmax(counts)
    # and finally return the most common label
    # print("Majority vote is: ", unique_labels[i])
    return unique_labels[i]


def get_error(predicted, actual):
    mask = (predicted == actual)
    return 1 - (np.sum(mask) / len(mask))

################ Recursive Learning Function ################
def learn(data, remaining_features):
    # pretty print the start
    counts_first = _train_labels.count(_unique_labels[0])
    counts_second = _train_labels.count(_unique_labels[1])

    print("[{} {} / {} {}]".format(counts_first, _unique_labels[0], counts_second, _unique_labels[1]))

    return dt_learn(data, remaining_features)


def dt_learn(data, remaining_features, current_level=0):
    """
    Returns a Node at the root of a learned Decision Tree (DT)

    Data should by a Numpy ndarray
    """
    # print("Enter learning...")
    # Check that we haven't exceeded max depth...
    if(current_level == MAX_DEPTH):
        # print("Max depth!")
        return Leaf(get_majority(data[:,-1]))   # reached max depth, use majority

    # Check for remaining features...
    elif (len(remaining_features) < 1):             # if we can't split anymore, just
        # print("No features left!", current_level)
        return Leaf(get_majority(data[:,-1]))       # use majority vote here

    # Otherwise we can maybe learn from this data!
    best_feature = None
    best_score = -1
    best_idx = -1
    feature_idx = -1
    for i, f in enumerate(remaining_features):                    # for each of the unused features
        true_idx = _attributes[f]                        # get the true index for feature
        current_feature_data = data[:, true_idx]         # retrieve the vector of data
        current_labels = data[:, -1]                # and the corresponding labels

        # Now get mutual info and see if it is higher than previous max
        m_i = mutual_info(current_labels, current_feature_data)
        # print("Mutual info for", f, ": ", m_i)
        if  m_i > best_score:
            best_feature = f
            best_score = m_i
            best_idx = true_idx
            feature_idx = i

    # Check for positive Mutual Information...
    if (best_score <= 0.0):
        # print("No MI left!", current_level)
        return Leaf(get_majority(data[:,-1]))      # majority vote

    # Now we've found the best feature to split on, we split into partitions:
    #   - One subset for which the chosen feature is the first option
    #   - One subset for the rest
    # print(best_feature)

    best_feature_data = data[:, best_idx]       # recover the column
    vals = sorted(list(set(best_feature_data))) # get unique values
    val_1, val_2 = vals[0], vals[1]

    # print(vals)
    # here we create a map to specify which way is which
    directions = {val_1 : 'left', val_2 : 'right'}

    partition_1 = data[best_feature_data == val_1]
    partition_2 = data[best_feature_data == val_2]

    # print(partition_1)
    # print(partition_2)

    remaining_features.pop(feature_idx)            # remove used feature

    # pretty print
    part_1 = list(partition_1[:,-1])
    part_2 = list(partition_2[:,-1])
    # elems_1, counts_1 = np.unique(partition_1[:,-1], return_counts=True)
    # elems_2, counts_2 = np.unique(partition_2[:,-1], return_counts=True)

    # print(part_1)
    # print(part_2)

    counts_1_first = part_1.count(_unique_labels[0])
    counts_1_second = part_1.count(_unique_labels[1])
    counts_2_first = part_2.count(_unique_labels[0])
    counts_2_second = part_2.count(_unique_labels[1])


    tail_1 =  "[{} {} / {} {}]".format(counts_1_first, _unique_labels[0], counts_1_second, _unique_labels[1])
    tail_2 =  "[{} {} / {} {}]".format(counts_2_first, _unique_labels[0], counts_2_second, _unique_labels[1])

    head_1 = best_feature + " = " + val_1 + ": "
    head_2 = best_feature + " = " + val_2 + ": "


    print("| " * (current_level+1) + head_1 + tail_1)
    left = dt_learn(partition_1, remaining_features[:], current_level+1)
    print("| " * (current_level+1) + head_2 + tail_2)
    right = dt_learn(partition_2, remaining_features[:], current_level+1)

    this_node = Node(best_feature, directions, left, right)

    # print("Made a node!")
    return this_node



################ Prediction functions ################
def predict(d_tree, data):
    """
    This one handles assembling the predictions and calling
    the error-calculating functions.

    Assumes top row of column headers is already removed.
    """
    labels = []
    data = np.array(data)
    # print(data)

    for row in data:
        pred = dt_predict(d_tree, row)
        labels.append(pred)

    assert len(labels) == len(data)
    # print(labels)

    return labels



def dt_predict(d_tree, sample):
    # We'll assume that the data is cleaned of feature names on the first row
    # print(sample)
    if isinstance(d_tree, Leaf):
        # print("*** A prediction!", d_tree.prediction, "***")
        return d_tree.prediction
    else:           # must be a node
        # get the feature of the node
        f_name = d_tree.att_name
        # print("Att name:", f_name)

        # get the master index of that feature
        f_index = _attributes[f_name]
        # print("Index:", f_index)

        # get the value of the data point at that feature
        data_f = sample[f_index]
        # print("Val:", data_f)

        # get the corresponding directions from the Node
        dir = d_tree.directions[data_f]
        # print("Direction:", dir)

        # now return the appropriate Node
        if dir == 'left':
            return dt_predict(d_tree.left, sample)
        elif dir == 'right':
            return dt_predict(d_tree.right, sample)

        print("Oh no!", sample)
        return None


################ I/O functions ################
def read_training_data(train_path):
    with open(train_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        train_data = list(reader)

    # get the attributes of the data -- should be the first row
    atts = train_data.pop(0)
    atts.pop(-1)                # remove the label as an attribute
    global _attributes
    _attributes = dict(zip(atts, range(len(atts))))

    # get labels, which are the last value of every row
    global _train_labels
    _train_labels = [row[-1] for row in train_data]
    global _unique_labels
    _unique_labels = np.unique(_train_labels)

    # return everything except the column headers
    return train_data


def read_test_data(test_path):
    with open(test_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        test_data = list(reader)

    # get the attributes of the data -- should be the first row
    atts = test_data.pop(0)

    # get labels, which are the last value of every row
    global _test_labels
    _test_labels = [row[-1] for row in test_data]

    # return everything except the column headers
    return test_data


def write_labels(labels, path):
    with open(path, 'w') as f_out:
        for label in labels:
            f_out.write(str(label) + "\n")


def write_metrics(path, train, test):
    with open(path, 'w') as f_out:
        tr = "error(train): " + str(train) + "\n"
        te = "error(test): " + str(test) + "\n"
        f_out.write(tr)
        f_out.write(te)



################ Main driver function ################
def main():
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    global MAX_DEPTH
    MAX_DEPTH = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    # read in data and get labels
    train_data = np.array(read_training_data(train_in))

    #******** training first!
    learned_tree = learn(train_data.copy(), list(_attributes))

    #********
    # now we've trained, try out our tree on the training data
    train_predictions = predict(learned_tree, train_data)

    # write the predictions to file
    write_labels(train_predictions, train_out)

    #******** do the same for test data
    test_data = np.array(read_test_data(test_in))
    test_predictions = predict(learned_tree, test_data)
    write_labels(test_predictions, test_out)


    #******** get the metrics
    train_error = get_error(np.array(train_predictions), np.array(_train_labels))
    test_error = get_error(np.array(test_predictions), np.array(_test_labels))
    write_metrics(metrics_out, train_error, test_error)






if __name__ == '__main__':
    if len(sys.argv) == 7:
        main()

    else:
        print("Wrong number of arguments!\n"
              + "Correct usage is 'python decisionTree.py <args... (6 args)>'")


  # get a column = [row[col_idx] for row in data[rows_desired]]

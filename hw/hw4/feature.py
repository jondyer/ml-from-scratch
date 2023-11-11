"""
Intro to Machine Learning
@author: Jonathan Dyer (jbdyer)
"""
import math
import csv
import sys
from collections import Counter

############################## Global Variables ################################
_vocab = {}

############################## Model 1 #########################################
def model_1(data):
    """
    Transforms the data according to model 1 (bag of words), which simply
    maps each word that appears in a given data point to 1 (as long as it's in
    the vocabulary).

    Parameters
    ----------
    data : matrix
        List of tokenized feature (word) vectors.

    Returns
    -------
    list of sets
        Each set mapping {word_in_vector : 1}
    """
    list_of_sets = []   # must maintain order

    for sample in data:
        raw = set(sample)                   # retrieve unique words in this sample
        words = raw.intersection(_vocab)    # only keep the ones in the vocab
        list_of_sets.append(words)          # add to our list

    return list_of_sets

############################## Model 2 #########################################
def model_2(data):
    """
    Transforms the data according to model 2 (trimmed bag of words), which
    maps each word that appears in a given data point to 1 (as long as it's in
    the vocabulary) as long as it appears less than THRESHOLD times.

    Parameters
    ----------
    data : matrix
        List of tokenized feature (word) vectors.

    Returns
    -------
    list of sets
        Each set mapping {word_in_vector : 1}
    """
    THRESHOLD = 4
    list_of_sets = []   # must maintain order

    for sample in data:
        count = Counter(sample)
        raw = set([key if val < THRESHOLD else None for key, val in count.items()])
        words = raw.intersection(_vocab)    # only keep the ones in the vocab
        list_of_sets.append(words)          # add to our list

    return list_of_sets

############################## File I/O ########################################
def read_vocab(filename):
    with open(filename, 'r') as file:
        lines = map(lambda x : x.split(), file.readlines())
        return {row[0] : int(row[1]) for row in lines}

def read_data(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    X = [line.split() for line in lines]
    y = [int(line.pop(0)) for line in X]

    return y, X

def write_formatted_data(data, labels, filename):
    with open(filename, 'w') as file:
        for idx, row in enumerate(data):
            string = str(labels[idx]) + '\t' + '\t'.join([str(_vocab[word]) + ':1' for word in row]) + '\n'

            file.write(string)


############################## MAIN function ###################################
def main():
    train_in = sys.argv[1]
    valid_in = sys.argv[2]
    test_in = sys.argv[3]
    dict_in = sys.argv[4]
    train_out = sys.argv[5]
    valid_out = sys.argv[6]
    test_out = sys.argv[7]
    feature_flag = sys.argv[8]


    ########################### PART 1: Read in data ###########################
    # 1a: Read in the dictionary
    global _vocab
    _vocab = read_vocab(dict_in)

    # 1b: Read in the training, validation, and test data
    train_labels, train_data = read_data(train_in)
    valid_labels, valid_data = read_data(valid_in)
    test_labels, test_data = read_data(test_in)

    ########################### PART 2: Transform data #########################
    if (int(feature_flag)==1):
        train_model = model_1(train_data)
        valid_model = model_1(valid_data)
        test_model = model_1(test_data)
    elif (int(feature_flag)==2):
        train_model = model_2(train_data)
        valid_model = model_2(valid_data)
        test_model = model_2(test_data)
    else:
        print("Invalid model selection! " + feature_flag + " Please provide <feature_flag> as 1 or 2\n")
        return 1

    ########################### PART 3: Write out data #########################
    write_formatted_data(train_model, train_labels, train_out)
    write_formatted_data(valid_model, valid_labels, valid_out)
    write_formatted_data(test_model, test_labels, test_out)




if __name__ == '__main__':
    if len(sys.argv) == 9:
        main()
    else:
        print("Wrong number of arguments!\n"
              + "Correct usage is 'python feature.py <args... (8 args)>'")

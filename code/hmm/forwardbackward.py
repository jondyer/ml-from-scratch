# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:09:09 2019

@author: jondyer
"""
import sys
import numpy as np


############################## Global Variables ################################
word_to_index = {}
tag_to_index = {}
index_to_word = {}
index_to_tag = {}


############################## File I/O #######################################
def read_dict(filename):
    with open(filename, 'r') as file:
        lines = list(file)
        dict1, dict2 = {word.strip() : idx for idx, word in enumerate(lines)}, {idx : word.strip() for idx, word in enumerate(lines)} 
        return dict1, dict2
    
def read_prior(filename):
    with open(filename, 'r') as file:
        lines = list(map(float, list(file)))
        return np.array(lines)

def read_trans(filename):
    with open(filename, 'r') as file:
        lines = [list(map(float, row.split())) for row in list(file)]
        return np.array(lines)
            
def read_emit(filename):
    with open(filename, 'r') as file:
        lines = [list(map(float, row.split())) for row in list(file)]
        return np.array(lines)

def read_data(filename):
    with open(filename, 'r') as file:
        lines = [elem.split() for elem in list(file)]
        
        # get all pairs in convenient tuple form
        all_pairs = [[tuple(item.split('_')) for item in row] for row in lines]

        return all_pairs

def write_predictions(filename, preds):
    with open(filename, 'w') as file:
        for x in preds:
            file.write(' '.join(list(map(lambda z: '_'.join(z), x))))
            file.write('\n')
            
def write_metrics(filename, acc, av):
    with open(filename, 'w') as file:
        file.write('Average Log-Likelihood: ' + str(av) + '\n')
        file.write('Accuracy: ' + str(acc) + '\n')


############################## Inference/Math #################################
def forward(pi, a, b, x):
    """
    pi = prior
    a = transition
    b = emission
    x = test data sequence (one data point/sentence)
    """
    global word_to_index, tag_to_index
    num_tags = len(tag_to_index)
    num_words = len(x)
    
    # set up alpha matrix
    alpha = np.zeros((num_tags, num_words))
    
    # now get alpha first column
    x_0 = word_to_index[x[0][0]]
    alpha[:,0] = b[:,x_0] * pi

    # and iterate over remaining columns/states
    t = 1
    while t < num_words:
        x_t = word_to_index[x[t][0]]
        alpha[:,t] = b[:,x_t] * (a.transpose() @ alpha[:,t-1])
        t += 1
        
    return alpha


def backward(a, b, x):
    """
    a = transition
    b = emission
    x = test data sequence (one data point/sentence)
    """
    global word_to_index, tag_to_index
    num_tags = len(tag_to_index)
    num_words = len(x)
    
    # set up alpha matrix
    beta = np.zeros((num_tags, num_words))
    
    # now get beta last column
    beta[:,-1] = 1

    # and iterate over remaining columns/states
    t = num_words - 2
    while t > -1:
        x_t = word_to_index[x[t+1][0]]
        beta[:,t] = a @ (b[:,x_t] * beta[:,t+1])
        t -= 1
        
    return beta


# %%
############################## Main method ####################################
def main():
    test_in = sys.argv[1]
    index_word_map = sys.argv[2]
    index_tag_map = sys.argv[3]
    prior_in = sys.argv[4]
    emit_in = sys.argv[5]
    trans_in = sys.argv[6]
    predict_out = sys.argv[7]
    metric_out = sys.argv[8]
    
    # First read in the dictionaries
    # pull global vars into scope
    global tag_to_index, index_to_tag, word_to_index, index_to_word
    tag_to_index, index_to_tag = read_dict(index_tag_map)
    word_to_index, index_to_word = read_dict(index_word_map)
    
    # Now get the prior, transition, and emission matrices as numpy arrays
    hmmprior = read_prior(prior_in)
    hmmtrans = read_trans(trans_in)
    hmmemit = read_emit(emit_in)
    
    # Read the test data
    test_data = read_data(test_in)
    
    # For every value in the test_data, do inference
    predictions = []
    total_predictions = 0
    total_correct = 0
    total_log_likelihood = 0.0
    for x in test_data:
        # get alpha and beta
        alpha = forward(hmmprior, hmmtrans, hmmemit, x)
        beta = backward(hmmtrans, hmmemit, x)
        
        # calculate conditional probs, then convert to tags
        cond_probs = alpha * beta
        predicted_indices = np.argmax(cond_probs, axis=0)
        predicted_tags = list(map(lambda x : index_to_tag[x], predicted_indices))
    
        # now add the predictions to our list
        words, tags = zip(*x)
        pred_list = list(zip(words, predicted_tags))
        predictions.append(pred_list)
    
        # update correct count, total count, and log likelihood
        npx = np.array(x)
        npp = np.array(pred_list)
        
        total_correct += sum(npx == npp)[1]
        total_predictions += len(x)
        if alpha[:,-1].sum() != 0:
            total_log_likelihood += np.log(alpha[:,-1].sum())
        
    # calculate accuracy and average log likelihood
    accuracy = total_correct / total_predictions
    avg_log_likelihood = total_log_likelihood / len(test_data)
    
    # now write out predictions and metrics
    write_predictions(predict_out, predictions)
    write_metrics(metric_out, accuracy, avg_log_likelihood)

if __name__ == '__main__':
    if len(sys.argv) == 9:
        main()
    else:
        print("Wrong number of arguments!\n"
              + "Correct usage is 'python forwardbackward.py <args... (8 args)>'")
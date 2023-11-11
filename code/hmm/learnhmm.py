# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:08:19 2019

@author: jondyer
"""
import sys
import numpy as np
from collections import Counter


############################## Global Variables ################################
word_to_index = {}
tag_to_index = {}
index_to_word = {}
index_to_tag = {}
num_sequences = 10000


#%%
############################## File I/O ########################################
def read_dict(filename):
    with open(filename, 'r') as file:
        lines = list(file)
        dict1, dict2 = {word.strip() : idx for idx, word in enumerate(lines)}, {idx : word.strip() for idx, word in enumerate(lines)} 
        return dict1, dict2

def write_prior(filename, prior):
    with open(filename, 'w') as file:
        for elem in prior:
            file.write("{:.18e}\n".format(elem))

def write_trans(filename, trans):
    with open(filename, 'w') as file:
        for row in trans:
            pr_row = list(map(lambda x : "{:.18e}".format(x), row))
            file.write(' '.join(pr_row) + "\n")
            
def write_emit(filename, emit):
    with open(filename, 'w') as file:
        for row in emit:
            pr_row = list(map(lambda x : "{:.18e}".format(x), row))
            file.write(' '.join(pr_row) + "\n")


############################## Learning ########################################
def prior(filename):
    global tag_to_index
    global index_to_tag
    
    with open(filename, 'r') as file:
        lines = [elem.split() for elem in list(file)][:num_sequences]
        
        # now we have the file as a list of lists, get the tags of first column
        first_column = [row[0] for row in lines]
        tags_first_column = [item.split('_')[1].strip() for item in first_column]
        
        # now set up result vector and fill for every entry in dictionary
        pi = np.zeros(len(tag_to_index))
        
        count_tags = Counter(tags_first_column)
        
        # add every count in the proper index of pi
        for idx in range(len(pi)):
            term = index_to_tag[idx]
            count = count_tags[term]
            pi[idx] = count + 1.0    # +1 for pseudocounts
            
        denom = len(tags_first_column) + len(pi)    # +1 for each class for pseudocounts
        
        # normalize by denominator
        pi = pi / denom
        
        return pi
        

def trans(filename):
    global tag_to_index
    global index_to_tag
    
    with open(filename, 'r') as file:
        lines = [elem.split() for elem in list(file)][:num_sequences]
        
        # get all tags so we can check transitions
        all_tags = [[item.split('_')[1].strip() for item in row] for row in lines]

        # now we'll use some clever Counter tricks to get bigram counts for each line
        a = np.ones((len(tag_to_index), len(tag_to_index)))     # start with ones for pseudocounts
        bigram_count = Counter()
        
        for row in all_tags:
            bigram_count.update(zip(row, row[1:]))
            
        # now we fill out the matrix
        for pair, ct in bigram_count.items():
            fst, scd = tag_to_index[pair[0]], tag_to_index[pair[1]]     # get indices
            a[fst, scd] += ct
            
        for row in a:
            row /= row.sum()
        
        return a


def emit(filename):
    global tag_to_index
    global index_to_tag
    
    with open(filename, 'r') as file:
        lines = [elem.split() for elem in list(file)][:num_sequences]
        
        # get all pairs so we can check emissions
        all_pairs = [[tuple(item.split('_')) for item in row] for row in lines]

        # now we'll use some clever Counter tricks to get counts of each pair
        # b is num_states(tags) X num_words
        b = np.ones((len(tag_to_index), len(word_to_index)))     # start with ones for pseudocounts
        pair_count = Counter()
        
        for row in all_pairs:
            pair_count.update(row)
            
        # now we fill out the matrix
        for pair, ct in pair_count.items():
            scd, fst = word_to_index[pair[0]], tag_to_index[pair[1]]     # get indices
            b[fst, scd] += ct
            
        for row in b:
            row /= row.sum()
        
        return b

# %%
############################## Empirical Qs ###################################
#train_in = "fulldata/trainwords.txt"
#prior_out = "hmmprior.txt"
#emit_out = "hmmemit.txt"
#trans_out = "hmmtrans.txt"
#
#index_word_map = "fulldata/index_to_word.txt"
#index_tag_map = "fulldata/index_to_tag.txt"
#
## First read in the dictionaries
#tag_to_index, index_to_tag = read_dict(index_tag_map)
#word_to_index, index_to_word = read_dict(index_word_map)
#
#
#hmmprior = prior(train_in)
#
## And the transition matrix
#hmmtrans = trans(train_in)
#
## Now the emission matrix
#hmmemit = emit(train_in)
#
#
## Finally write them all out
#write_prior(prior_out, hmmprior)
#write_trans(trans_out, hmmtrans)
#write_emit(emit_out, hmmemit)
#
#
## %%
#import matplotlib.pyplot as plt
#plt.figure(figsize=(10,5))
#
#x = [10,100,1000,10000]
#tr = [-122.474, -110.255, -101.0149, -95.384]
#te = [-130.865, -115.858, -105.308, -98.528]
#
## first the training
#plt.plot(x, tr, linestyle='-', alpha=0.5, color='C0', label = 'Train avg log-likelihood')
#plt.plot(x, te, linestyle='-', alpha=0.5, color='C1', label = 'Test avg log-likelihood')
#
#plt.legend(title = 'Prediction data set', loc = 'best')
#plt.title("Avg log-likelihood by number of learning sequences for HMM")
#plt.xlabel('# sequences used for learning')
#plt.ylabel('Avg log-likelihood for all sequences')
#plt.savefig('plot.png')
#plt.show()


# %%
############################## Main method ###################################
def main():
    train_in = sys.argv[1]
    index_word_map = sys.argv[2]
    index_tag_map = sys.argv[3]
    prior_out = sys.argv[4]
    emit_out = sys.argv[5]
    trans_out = sys.argv[6]
    
    # pull global vars into scope
    global tag_to_index, index_to_tag, word_to_index, index_to_word
    
    # First read in the dictionaries
    tag_to_index, index_to_tag = read_dict(index_tag_map)
    word_to_index, index_to_word = read_dict(index_word_map)
    
    # Now get the prior
    hmmprior = prior(train_in)

    # And the transition matrix
    hmmtrans = trans(train_in)
    
    # Now the emission matrix
    hmmemit = emit(train_in)
    
    
    # Finally write them all out
    write_prior(prior_out, hmmprior)
    write_trans(trans_out, hmmtrans)
    write_emit(emit_out, hmmemit)
    

if __name__ == '__main__':
    if len(sys.argv) == 7:
        main()
    else:
        print("Wrong number of arguments!\n"
              + "Correct usage is 'python learnhmm.py <args... (6 args)>'")

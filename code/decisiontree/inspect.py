"""
Intro to Machine Learning
@author: Jonathan Dyer (jbdyer)
"""
import math
import csv
import sys
from collections import Counter

# driver method for inspecting the input file
def inspect(infile, outfile):
    with open(infile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        data = list(reader)

    # get the labels of the data
    labels = [row[-1] for row in data[1:]]

    # now calculate entropy
    ent = entropy(labels)
    print(ent)

    # now get error
    err = error(labels)
    print(err)

    with open(outfile, 'w') as f_out:
        f_out.write('entropy: ' + str(ent) + '\n')
        f_out.write('error: ' + str(err))


def entropy(data):
    total = 0
    for i in set(data):     # for each distinct value of Y
        p = prob(i, data)   # get probability of Y taking that value
        log_p = math.log2(p)
        total += (p * log_p)

    return -1 * total


def prob(val, data):
    return data.count(val) / len(data)


def error(data):
    count = Counter(data)
    majority = count.most_common(1)[0][1]   # gives a tuple, just want value
    total = sum(count.values())
    return (total - majority) / total


if __name__ == '__main__':
    if len(sys.argv) == 3:
        infile = sys.argv[1]
        outfile = sys.argv[2]

        inspect(infile, outfile)
    else:
        print("Wrong number of arguments!\n"
              + "Correct usage is 'python inspect.py <input> <output>'")

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 19:48:13 2019

@author: jondyer
"""

import sys


def my_reverse(infile, outfile):
    with open(infile, 'r') as f:
        data = f.readlines()
    
    data_reversed = reversed(data)
    
    with open(outfile, 'w') as f_out:
        f_out.writelines(data_reversed)


if __name__ == '__main__':
    if len(sys.argv) == 3:
        infile = sys.argv[1]
        outfile = sys.argv[2]
    
        my_reverse(infile, outfile)
    else:
        print("Wrong number of arguments!\n"
              + "Correct usage is 'python reverse.py [input_file] [output_file]'")
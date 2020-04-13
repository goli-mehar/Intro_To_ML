import csv
import numpy as np
import math
import sys
import copy

""" ---------------------------------------------------------------------------
                     Initialize Data to Read In
--------------------------------------------------------------------------- """
""" 
@brief Reads in test data and creates the corresponding matrix object

@param[in] file_path - path to read data from
@return data - matrix of data
"""
def initialize_train_data(file_path):
    data = []
    sentence = []

    with open(file_path) as f:
        reader = f.read()

        for row in reader.split('\r\n'):
            sentence = str(row).split(' ')
            sent_list = []

            for word in sentence:
                elems = word.split("_")
                sent_list.append(elems)

            data.append(sent_list)

    return data

""" 
@brief Reads in word/tag reference list into correpsonding matrix object

@param[in] file_path - path to read from
@return reference - matrix of values
"""
def initialize_reference(file_path):
    reference = []

    with open(file_path) as f:
        reader = f.read()
        reference = reader.split('\r\n')
        
    return reference

""" ---------------------------------------------------------------------------
                        Matrix Index Determinations
--------------------------------------------------------------------------- """
""" Given a word, finds corresponding matrix indexes

@param[in] word_pair - pair of x and y values
@param[in] X - x values index reference
@param[in] Y - y values index reference
"""
def find_index(word_pair, X, Y):

    #x, y = word_pair

    #x_i = numpy.argwhere(X)
    #y_i = numpy.argwhere(Y)

    #return x_i, y_
    return

""" ---------------------------------------------------------------------------
                            Parameter Updates
--------------------------------------------------------------------------- """

"""
@brief Increments Matrix Values

@param[in] mat - prior matrix
@param[in] j - row index of update cell
@param[in] k - column index of update cell
"""
def update_matrix(mat, j, k):
    return
""" ---------------------------------------------------------------------------
                            Matrix Normalization
--------------------------------------------------------------------------- """
"""
@brief: Given a row vector, normalizes values in row

@param[in] row
@param[out] normalized row
"""
def normalize_row(row):
    return

"""
@brief: Given a matrix, normalizes value across each row

@param[in] matrix
@param[out] normalized matrix
"""
def normalize_matrix(mat):
    return
""" ---------------------------------------------------------------------------
                            Printing Helpers
--------------------------------------------------------------------------- """

"""
@brief: Prints a given matrix to given outfile

Format: Row values seperated by spaces, with eol terminated by a newline
character

@param[in] mat - matrix with values to print
@param[in] file - file to print to
"""
def print_matrix(mat, outfile):
    return
""" ---------------------------------------------------------------------------
                            Main Run Routines
--------------------------------------------------------------------------- """
"""
@brief: Initializes needed data for program procedure

Reads in input data set, X and Y index references
Also initializes empty prior matrix, transition matrix and
emission matrix 

@param[in] data - object containing all elements
"""
def initialize(data):

    data.train_data = initialize_train_data(data.train_path)
    data.word_ref = initialize_reference(data.word_to_index_path)
    data.tag_ref = initialize_reference(data.tag_to_index_path)

    # train_data: read in train data
    # X = read in word to index data
    # Y = read in tag to index data

    # find G dimension
    #find M dimesnions

    # set up prior matrix (1XG)
    # set up transition matrix (GXG)
    # set up emmision matrix (G X M)

"""
@brief: Main program routine to learn HMM parameters
using the closed MLE solutions

Manually computes prior, transmission and emmision
matrices

@param[in] data - object containing all elements
"""
def learn_parameters(data):
    return

    #initialize sentence, iterator vals
    #emm_mat,trans_mat,prior_mat given by data object

    # run loop through sentence
    #loop through sentence, based on index

"""
@brief Prints paramter matrices to given output files

@param[in] data - object containing matrices and output paths
"""
def print_output(data):
    return
    #print emm_mat,trans_mat,prior_mat to corresponding systems


"""
@brief Main routine - runs learning procedure

Intializes data, learns parameters and prints them to
output files given by command line
"""
def main(data):
    initialize(data)
    learn_parameters(data)
    print_output(data)

if __name__ == "__main__":
    class Struct(object): pass
    data = Struct()

    data.train_path = sys.argv[1]
    data.word_to_index_path = sys.argv[2]
    data.tag_to_index_path = sys.argv[3]
    data.prior_outpath = sys.argv[4]
    data.emit_outpath = sys.argv[5]
    data.trans_outpath = sys.argv[6]

    main(data)

    

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
        reader = reader.replace("\r", "")

        i = 0
        for row in reader.split('\n'):

            if (i < 10):
                sentence = str(row).split(' ')
                sent_list = []

                for word in sentence:
                    elems = word.split("_")
                    if len(elems) == 2:
                        sent_list.append(elems)

                data.append(sent_list)
            i += 1
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
        reader = reader.replace("\r", "")
        reference = reader.split('\n')

        if "" in reference:
            reference.remove("")

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

    if(type(word_pair) == list and len(word_pair) == 2):
        x, y = word_pair

        x_i = X.index(x)
        y_i = Y.index(y)
    else:
        x_i = -1
        y_i = -1

    return x_i, y_i

""" ---------------------------------------------------------------------------
                            Parameter Updates
--------------------------------------------------------------------------- """

"""
@brief Increments Matrix Values

@param[in] mat - prior matrix
@param[in] j - row index of update cell
@param[in] k - column index of update cell
@return mat_new - updated matrix
"""
def update_matrix(mat, j, k):

    mat_new = copy.deepcopy(mat)
    if (j != -1 or k != -1):
        if (j == -1):
            mat_new[k] = mat_new[k] + 1
        else:
            mat_new[j][k] = mat_new[j][k] + 1

    return mat_new

""" ---------------------------------------------------------------------------
                            Matrix Normalization
--------------------------------------------------------------------------- """
"""
@brief: Given a row vector, normalizes values in row

@param[in] row
@param[out] norm_row normalized row
"""
def normalize_row(row):
    return row / np.sum(row)

"""
@brief: Given a matrix, normalizes value across each row

@param[in] matrix
@param[out] normalized matrix
"""
def normalize_matrix(mat):

    norm_mat = copy.deepcopy(mat)

    for i in range(0, len(norm_mat)):
        norm_mat[i] = normalize_row(norm_mat[i])

    return norm_mat

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

    f = open(outfile, "w")
    
    for row in mat:
        if type(row) == np.float64:
            f.write("%.18E" % (row))
        else:
            for i in range(0, len(row)):
                if (i == len(row) - 1):
                    f.write("%.18E" % (row[i]))
                else:
                    f.write("%.18E " % (row[i]))
        f.write("\n")
    
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

    G = len(data.tag_ref) #Number of possible tags
    M = len(data.word_ref) #Number of possible words

    #initialize with a +1 pseudocount
    data.prior_mat = np.ones((G, 1))
    data.trans_mat = np.ones((G, G))
    data.emit_mat = np.ones((G, M))

"""
@brief: Main program routine to learn HMM parameters
using the closed MLE solutions

Manually computes prior, transmission and emmision
matrices

@param[in] data - object containing all elements
"""
def learn_parameters(data):

    cur_x = ""
    cur_y = ""
    prev_y = ""

    for sentence in data.train_data:
        for i in range(0, len(sentence)):
            word_pair = sentence[i]

            cur_x, cur_y = find_index(word_pair, data.word_ref, data.tag_ref)

            if (i == 0):
                data.prior_mat = update_matrix(data.prior_mat, -1, cur_y)
            else:
                data.trans_mat = update_matrix(data.trans_mat, prev_y, cur_y)
            
            data.emit_mat = update_matrix(data.emit_mat, cur_y, cur_x)
            prev_y = cur_y
    
    data.prior_mat = normalize_row(data.prior_mat)
    data.emit_mat = normalize_matrix(data.emit_mat)
    data.trans_mat = normalize_matrix(data.trans_mat)
    return

"""
@brief Prints paramter matrices to given output files

@param[in] data - object containing matrices and output paths
"""
def print_output(data):

    print_matrix(data.prior_mat, data.prior_outpath)
    print_matrix(data.emit_mat, data.emit_outpath)
    print_matrix(data.trans_mat, data.trans_outpath)

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

    

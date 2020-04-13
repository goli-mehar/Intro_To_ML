
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

""" ---------------------------------------------------------------------------
                            Matrix Normalization
--------------------------------------------------------------------------- """

"""
@brief: Given a row vector, normalizes values in row

@param[in] row
@param[out] normalized row
"""
def normalize_row(row):

"""
@brief: Given a matrix, normalizes value across each row

@param[in] matrix
@param[out] normalized matrix
"""
def normalize_matrix(mat):

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

""" ---------------------------------------------------------------------------
                            Main Run Routines
--------------------------------------------------------------------------- """
"""
@brief: Initializes needed data for program procedure

Reads in input data set, X and Y index references
Also initializes empty prior matrix (Pi), transition matrix (B) and
emission matrix (A)

@param[in] data - object containing all elements
"""
def initialize(data):

    # train_data: read in train data
    # X = read in word to index data
    # Y = read in tag to index data

    # find G dimension
    #find M dimesnions

    # Pi = set up prior matrix (1XG)
    # B = set up transition matrix (GXG)
    # A = set up emmision matrix (G X M)

"""
@brief: Main program routine to learn HMM parameters
using the closed MLE solutions

Manually computes prior, transmission and emmision
matrices

@param[in] data - object containing all elements
"""
def learn_parameters():

    #initialize sentence, iterator vals
    #A,B,pi given by data object

    # run loop through sentence
    #loop through sentence, based on index

"""
@brief Prints paramter matrices to given output files

@param[in] data - object containing matrices and output paths
"""
def print_output():

    #print pi, A, B to corresponding systems

"""
@brief Main routine - runs learning procedure

Intializes data, learns parameters and prints them to
output files given by command line
"""
def main():

    //create data struct

    #initialize values from cmd line
    #initialize in struct 
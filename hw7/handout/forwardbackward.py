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
        for row in reader.split('\n'):
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
        reader = reader.replace("\r", "")
        reference = reader.split('\n')

        print(reference)

    return reference

""" 
@brief Reads in probability matrix into correpsonding numpy object

@param[in] file_path - path to read from
@return reference - matrix of values
"""
def initialize_prob_matrix(file_path):
    mat = np.loadtxt(file_path, delimiter = " ")
    
    return mat
""" ---------------------------------------------------------------------------
                            Helper Functions
--------------------------------------------------------------------------- """

"""
@brief: Determines number of correct predictions

Counts how many predicted tag values in sentence match the actual tags

@param[in] Pred - predicted tag values
@param[in] sentence - list of actual word,tag tuples
@return count - the number of correct predictions
"""
def num_correct(pred, sentence):
    return

"""
@brief: Determines matrix index of given word/tag

@param[in] value - word/tag to search for
@parm[in] reference - reference with words/tags and indexes
@returns index of word/tag in reference
"""
def value_to_tag(value, reference):
    return

"""
@brief: Determines word/tag value of given matrix index

@param[in] index - index to search for
@parm[in] reference - reference with words/tags and indexes
@returns word/tag at index in reference
"""
def tag_to_value(value, reference):
    return

""" ---------------------------------------------------------------------------
                        Forward/Backward Routines
--------------------------------------------------------------------------- """
"""
@brief: Runs forward routines for HMM

@param[in] sentence - list of actual word,tag tuples
@param[in] emm_mat - emmission matrix
@param[in] trans_mat - transition matrix
@param[in] prior_mat - prior matrix
@return alpha - generated alpha matrix
"""
def forward(sentence, emm_mat, trans_mat, prior_mat):
    return

"""
@brief: Runs backward routines for HMM

@param[in] sentence - list of actual word,tag tuples
@param[in] emm_mat - emmission matrix
@param[in] trans_mat - transition matrix
@return beta - generated beta matrix
"""
def backward(sentence, emm_mat, trans_mat):
    return
""" ---------------------------------------------------------------------------
                        Log-Likelihood Calculations
--------------------------------------------------------------------------- """
"""
@brief: computes log-likelihood of a given sentence

The log-likelihood of a given sentence (set of (word, tag) values) can be found
in a hidden markov model by multiplying the calculated alphas of the
lawst word in each set.

@param[in] max_alphas - alpha values associated with last word of each sentence
@return l - calculated log-likelihood
"""
def log_likelihood(max_alphas):
    return
""" ---------------------------------------------------------------------------
                        Prediction Routines
--------------------------------------------------------------------------- """
"""
@brief: Predicts most likely tag using a maximum bayes likelihood estimator

Takes tag with highest lieklihood

@param[in] prob_mat - matrix of probabilities
@param[in] tag_ref - matrix of tag index references
@return pred - predicted values
"""
def predict(prob_mat, tag_ref):
    return

""" ---------------------------------------------------------------------------
                            Print Routines
--------------------------------------------------------------------------- """

"""
@brief: Writes predicted tags to output file

@param[in] pred - predicted values
@param[in] sentence - list of actual word,tag tuples
@param[in] outfile - file to write to
"""
def write_pred(pred, sentence, outfile):
    return

"""
@brief: Writes average log-likelyhood and error rate to given output file

@param[in] avg_likelihood - average likelihood
@param[in] error - error rate
@param[in] outfile - file to write to
"""
def write_metrics(avg_likelihood, error, outfile):
    return

""" ---------------------------------------------------------------------------
                            Main Run Routines
--------------------------------------------------------------------------- """
"""
@brief: Initializes needed data for program procedure

Reads in input data set, word and tag index references
Also initializes prior matrix, transition matrix and
emission matrix from inputed files

@param[in] data - object containing all elements
"""
def initialize(data):

    data.test_data = initialize_train_data(data.test_inpath)
    data.word_ref = initialize_reference(data.word_to_index_inpath)
    data.tag_ref = initialize_reference(data.tag_to_index_inpath)

    data.prior_mat = initialize_prob_matrix(data.prior_inpath)
    data.emit_mat = initialize_prob_matrix(data.emit_inpath)
    data.trans_mat = initialize_prob_matrix(data.trans_inpath)
# Read in A, B, pi
# Read in X, Y, test data

"""
@brief: Runs main forward backward algorithm

@param[in] data - object containing all elements
"""
def run_forward_backward_alg(data):
    return

""" ---------------------------------------------------------------------------
                                    Main
--------------------------------------------------------------------------- """

"""
@brief Main routine - runs learning procedure

Intializes data, learns parameters and prints them to
output files given by command line
"""
def main(data):
    initialize(data)
    run_forward_backward_alg(data)

if __name__ == "__main__":
    class Struct(object): pass
    data = Struct()

    data.test_inpath = sys.argv[1]
    data.word_to_index_inpath = sys.argv[2]
    data.tag_to_index_inpath = sys.argv[3]
    data.prior_inpath = sys.argv[4]
    data.emit_inpath = sys.argv[5]
    data.trans_inpath = sys.argv[6]
    data.predictions_outpath = sys.argv[7]
    data.metrics_outpath = sys.argv[8]

    main(data)
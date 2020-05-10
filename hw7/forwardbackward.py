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
                if len(elems) == 2:
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

        if "" in reference:
            reference.remove("")

    return reference

""" 
@brief Reads in probability matrix into correpsonding numpy object

@param[in] file_path - path to read from
@return reference - matrix of values
"""
def initialize_prob_matrix(file_path):
    mat = np.loadtxt(file_path, delimiter = " ")
    
    return np.transpose(mat)
""" 
@brief Clears potential prediction data from previous run

@param[in] file_path - path to clear
"""
def clear_predictions(file_path):
    f = open(file_path, "w")
    f.close
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
def count_correct(pred, sentence):
    count = 0

    for i in range(0, len(pred)):
        if pred[i] == sentence[i][1]: #actual y tag is second elem in tuple
            count += 1

    return count

"""
@brief: Determines matrix index of given word/tag

@param[in] value - word/tag to search for
@parm[in] reference - reference with words/tags and indexes
@returns index of word/tag in reference
"""
def value_to_index(value, reference):
    i = reference.index(value)
    return i

"""
@brief: Determines word/tag value of given matrix index

@param[in] index - index to search for
@parm[in] reference - reference with words/tags and indexes
@returns word/tag at index in reference
"""
def index_to_value(index, reference):
    v = reference[index]
    return v

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
def forward(alpha, sentence, emit_mat, trans_mat, prior_mat, word_ref):
    m = value_to_index(sentence[0][0], word_ref)
    alpha[0] = np.multiply(prior_mat, emit_mat[m])

    for t in range(1, len(sentence)):
        m = value_to_index(sentence[t][0], word_ref)
        alpha[t] = np.matmul(trans_mat, alpha[t-1])
        alpha[t] = np.multiply(alpha[t], emit_mat[m])

    return alpha

"""
@brief: Runs backward routines for HMM

@param[in] sentence - list of actual word,tag tuples
@param[in] emm_mat - emmission matrix
@param[in] trans_mat - transition matrix
@return beta - generated beta matrix
"""
def backward(beta, sentence, emit_mat, trans_mat, word_ref):
    T = len(sentence)

    m = value_to_index(sentence[T-1][0], word_ref)
    beta[T-1] = 1

    for t in range(T-2, -1, -1):
        m = value_to_index(sentence[t+1][0], word_ref)
        beta[t] = np.multiply(beta[t+1], emit_mat[m])
        beta[t] = np.matmul(beta[t], trans_mat)
    return beta
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
    sum_vals = sum(max_alphas)
    
    if(sum_vals) != 0:
        l = math.log(sum_vals)
    else:
        l = 0
    return l
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

    pred = []

    for t in range(0, len(prob_mat)):
        i = np.argmax(prob_mat[t])
        y_hat = index_to_value(i, tag_ref)
        pred.append(y_hat)

    return pred

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
    f = open(outfile, "a")

    for t in range(0, len(pred)):
        f.write("%s_%s" % (sentence[t][0], pred[t]))

        if (t < len(pred) - 1):
            f.write(" ")
    f.write("\n")
    f.close()
    return

"""
@brief: Writes average log-likelyhood and error rate to given output file

@param[in] avg_likelihood - average likelihood
@param[in] error - error rate
@param[in] outfile - file to write to
"""
def write_metrics(avg_likelihood, error, outfile):
    f = open(outfile, "w")

    f.write("Average Log-Likelihood: %.16f\n" % (avg_likelihood))
    f.write("Accuracy: %.16f\n" % (error))
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

    data.M = len(data.word_ref) #Number of possible words
    data.G = len(data.tag_ref) #Number of possible tags
    data.N = len(data.test_data) #Number of sentences in set

    clear_predictions(data.predictions_outpath)

"""
@brief: Runs main forward backward algorithm

@param[in] data - object containing all elements
"""
def run_forward_backward_alg(data):
    total_log_likelihood = 0
    num_correct = 0
    total = 0

    for sentence in data.test_data:
        T  = len(sentence)
        alpha = np.zeros((T,data.G))
        beta = np.zeros((T,data.G))

        alpha = forward(alpha, sentence, data.emit_mat, data.trans_mat, data.prior_mat, data.word_ref)
        beta = backward(beta, sentence, data.emit_mat, data.trans_mat, data.word_ref)

        prob_mat = np.multiply(alpha, beta) #multiply elementwise
        total_log_likelihood += log_likelihood(alpha[T-1])

        pred = predict(prob_mat, data.tag_ref)
        write_pred(pred, sentence, data.predictions_outpath)
        num_correct += count_correct(pred, sentence)
        total += T

    average_log_likelihood = total_log_likelihood/data.N
    error = float(num_correct) / float(total)
    write_metrics(average_log_likelihood, error, data.metrics_outpath)
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
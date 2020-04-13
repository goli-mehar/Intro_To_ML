
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
def num_correct(pred, sentence)

"""
@brief: Determines matrix index of given word/tag

@param[in] value - word/tag to search for
@parm[in] reference - reference with words/tags and indexes
@returns index of word/tag in reference
"""
def value_to_tag(value, reference)

"""
@brief: Determines word/tag value of given matrix index

@param[in] index - index to search for
@parm[in] reference - reference with words/tags and indexes
@returns word/tag at index in reference
"""
def tag_to_value(value, reference)

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
def forward(sentence, emm_mat, trans_mat, prior_mat)

"""
@brief: Runs backward routines for HMM

@param[in] sentence - list of actual word,tag tuples
@param[in] emm_mat - emmission matrix
@param[in] trans_mat - transition matrix
@return beta - generated beta matrix
"""
def backward(sentence, emm_mat, trans_mat)
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
def log_likelihood(max_alphas)
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
def predict(prob_mat, tag_ref)

""" ---------------------------------------------------------------------------
                            Print Routines
--------------------------------------------------------------------------- """

"""
@brief: Writes predicted tags to output file

@param[in] pred - predicted values
@param[in] sentence - list of actual word,tag tuples
@param[in] outfile - file to write to
"""
def write_pred(pred, sentence, outfile)

"""
@brief: Writes average log-likelyhood and error rate to given output file

@param[in] avg_likelihood - average likelihood
@param[in] error - error rate
@param[in] outfile - file to write to
"""
def write_metrics(avg_likelihood, error, outfile)

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

# Read in A, B, pi
# Read in X, Y, test data

"""
@brief: Runs main forward backward algorithm

@param[in] data - object containing all elements
"""
def run_forward_backward_alg(data):

""" ---------------------------------------------------------------------------
                                    Main
--------------------------------------------------------------------------- """

"""
@brief Main routine - runs learning procedure

Intializes data, learns parameters and prints them to
output files given by command line
"""
def main():

    //create data struct

    #initialize values from cmd line
    #initialize in struct 
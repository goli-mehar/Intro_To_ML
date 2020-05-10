"""
Author: Mehar Goli <mgoli@andrew.cmu.edu>

This program runs a neural network learner for text letter evaluation.
This is done using gradient descent. 
The neuralnetwork in question consists of a one hidden layer
network with a sigmoid activation function on the hidden layer and softmax on
the output layer. The SGD learner works through the implementation
of the forward backward algorithm to run the neural netqork and then 
calculate the gradient, it also uses a minimized cross entropy objective function.

The program then also tests the learner on the original training dataset and then
a new testset, predictions, cross entropies and overall error are the written
to output files.

(Note: running command setup is based on that provided by course handout,
thus description are taken and modified from handout. Handout can be found in 
main folder.

To run, use the following command:

python neuralnet.py data_files/<train input> data_files/<test input> 
<train out> <test out> <metrics out> <num epoch>
<hidden units> <init flag> <learning rate>

1. <train input>: path to the training input .csv file
2. <test input>: path to the test input .csv file
3. <train out>: path to output .labels file to which the prediction on the training data should be
written
4. <test out>: path to output .labels file to which the prediction on the test data should be written

5. <metrics out>: path of the output .txt file to which metrics such as train and test error should
be written 
6. <num epoch>: integer specifying the number of times backpropogation loops through all of the
training data (e.g., if <num epoch> equals 5, then each training example will be used in backpropogation
5 times).
7. <hidden units>: positive integer specifying the number of hidden units.
8. <init flag>: integer taking value 1 or 2 that specifies whether to use RANDOM or ZERO initialization
(see Section 2.2.1 and Section 2.3)—that is, if init_flag==1 initialize your weights randomly
from a uniform distribution over the range [-0.1,0.1] (i.e. RANDOM), if init_flag==2 initialize all
weights to zero (i.e. ZERO). For both settings, always initialize bias terms to zero.
9. <learning rate>: float value specifying the learning rate for SGD.

The following datasets are provided:

smallTrain.csv/smallTest.csv
mediumTrain.csv/mediumTest.csv
largeTrain.csv/largeTest.csv
"""

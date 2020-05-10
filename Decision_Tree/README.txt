"""
Author: Mehar Goli <mgoli@andrew.cmu.edu>

This program runs a decision tree learner on a given training dataset consisting of a
list of datapoints with a list of attributes and a given classification.
The tree learns the optimal calssification for a given set of attrubute 
values by maximizing ginigain. After learning the tree, the tree is applied 
to the training set and a given test data set and outputed to a predictions file
The inspection.py files is then run to calculate gini impurity and error rate.

(Note: running command setup is based on that provided by course handout,
thus description are taken and modified from handout. Handout can be found in 
main folder.

To run, use the following command:

python decisionTree.py data_files/inputs/<train input> data_files/inputs/<test
input> <max depth> <train out> <test out> <metrics out>t.

1. <train input>: path to the training input .tsv file
2. <test input>: path to the test input .tsv file
3. <max depth>: maximum depth to which the tree should be built
4. <train out>: output .labels file for training data predictions
5. <test out>: output .labels file for test data  predictions
6. <metrics out>: output .txt file to for train and test error

To inspect data and find error, us the following command:

python inspection.py data_files/inputs/<input> <output>

1. <input> dataset inputted to decision tree learner
2. <ouput> .labels file of predictions determined by learner

The following datasets are provided:

education_train.tsv, education_test.tsv
politician_train.tsv, politician_test.tsv
small_train.tsv/small_test.tsv
"""

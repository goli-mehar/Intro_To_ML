Author: Mehar Goli <mgoli@andrew.cmu.edu>

This assignment implements a basic decision stump learner. Data for each data 
point consists of a list of attributes with varying possible values, and a final 
classification.The program goal takes in a given training set from which it 
determines the majority classification for each value of a given target 
attribute. It then takes in a test set, predicting the classifications of the 
test data points by looking at the target attribute, and assigning the majority 
classifcations learned earlier. Finally, a list of predicted values and error 
rate of the learner are relayed in the testOutput and metrics output 
respectively.

To use: command must follow the given format

python decisionStump.py data_files/<train input> data_files/<test input> \
<split index> <train out>.labels <test out>.labels <metrics out>

train input: training data 
test input: testing data
split index: index of target attribute
train out: a .labels output path for learner predictions on training data
test out: a .labels. output path for learner predictions on testing data

The following datasets are provided:
education_train.tsv, education_test.tsv
politician_train.tsv, politician_test.tsv

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
import csv
import numpy as np
import math
import sys
import copy

##### Initialization #######

def object_new(D,K,M):

    obj = Struct()
    obj.a = np.zeros((1,D))
    obj.z = np.zeros((D+1,1))
    obj.b = np.zeros((1,K))
    obj.y_hat = np.zeros((1, K))
    obj.j = 0

    return obj

def initialize(data):
#Create alpha, beta, a, b, z, y_hat matrixes
    
    
    data.M = data.trainData.shape[1] - 1
    data.K = 10
    data.N_train = len(data.trainData)
    data.N_test= len(data.testData)

    D, K, M = data.D, data.K, data.M

    if (data.initFlag == 1):
        data.alpha = np.random.uniform(low = -0.1, high = 0.1, size = (M+1, D))
        data.beta = np.random.uniform(low = -0.1, high = 0.1, size = (D+1, K))
    else:
        data.alpha = np.zeros((M+1, D))
        data.beta = np.zeros((D+1, K))

####### Propogation Modules ########

### LINEAR MODULE ###


def linear_forward(values, weights):
    result = np.matmul(np.transpose(weights), values)
    return result

def linear_backward(values, weights, grad):

    grad_weights = np.transpose(np.outer(grad, values))
    grad_values = np.matmul(weights, grad)

    return grad_weights, grad_values[0:grad_values.size-1]

### SIGMOID MODULE ###

def sigmoid_forward(a):
    x = -a
    z = 1 / (1 + np.exp(x))
    return z

def sigmoid_backward(a, z, grad_z):
    z_tmp = np.multiply(z, 1-z)
    grad_a = np.multiply(grad_z, z_tmp)
    return grad_a

### SOFTMAX MODULE ###

def softmax_forward(b):
    y = np.exp(b)
    y = y / np.sum(y)

    return y

def softmax_backward(b, y_hat, y):

    y_vect = np.zeros(y_hat.shape)
    y_vect[int(y)] = 1
    
    y_diff = y_hat - y_vect
    return y_diff

### CROSS ENTROPY MODULE ###

##log function###

def crossentropy_forward(y, y_hat):

    value = y_hat[int(y)]
    j = -math.log(value)
    
    return j

def mean_cross_entropy(dataset, obj, alpha, beta):
    N, cols = dataset.shape
    cross_entropy_J = 0
    for elem in dataset:
        y = elem[0]
        x = copy.deepcopy(elem[1:])

        obj = forward_propogation(obj, x, y, alpha, beta)
        cross_entropy_J += obj.j
    
    mean_entropy = cross_entropy_J/N

    return mean_entropy
####### Propogation Algorithms ########

def forward_propogation(obj, x, y, alpha, beta):
    x_nobias = copy.deepcopy(x)
    x_wbias = np.append(x_nobias, 1)

    obj.a = linear_forward(x_wbias, alpha)
    z_nobias = sigmoid_forward(obj.a)
    obj.z = np.append(z_nobias, 1)

    obj.b = linear_forward(obj.z, beta)
    obj.y_hat = softmax_forward(obj.b)
    obj.j = crossentropy_forward(y, obj.y_hat)

    return obj

def backward_propogation(obj, x, y, alpha, beta, i):
    x_nobias = copy.deepcopy(x)
    x_wbias = np.append(x_nobias, 1)

    grad_b = softmax_backward(obj.b, obj.y_hat, y)

    z_nobias = obj.z[0:obj.z.size-1]
    grad_beta, grad_z = linear_backward(obj.z, beta, grad_b)
    grad_a = sigmoid_backward(obj.a, z_nobias, grad_z)
    grad_alpha, grad_x = linear_backward(x_wbias, alpha, grad_a)

    return grad_alpha, grad_beta


######## Training Functions ########
def sgd_run(dataset, obj, alpha, beta, learn_rate):
    
    i = 0
    for elem in dataset:
        y = elem[0]
        x = copy.deepcopy(elem[1:])
        obj = forward_propogation(obj, x, y, alpha, beta)

        grad_alpha, grad_beta = backward_propogation(obj, x, y, alpha, beta, i)

        alpha = alpha - (learn_rate*grad_alpha)
        beta = beta - (learn_rate*grad_beta)

        i+=1
    return obj, alpha, beta

def train(data):

    obj = object_new(data.D,data.K,data.M)

    train_ent = []
    test_ent = []
    e = 0
    while (e < data.E):
        obj, data.alpha, data.beta = sgd_run(data.trainData, obj, data.alpha, data.beta, data.learnRate)
        
        train_mean_crossentropy = mean_cross_entropy(data.trainData, obj, data.alpha, data.beta)
        test_mean_crossentropy = mean_cross_entropy(data.testData, obj, data.alpha, data.beta)

        train_ent.append(train_mean_crossentropy)
        test_ent.append(test_mean_crossentropy)
        writeLossToMetrics(e, train_mean_crossentropy, "train", data.metricsOutPath)
        writeLossToMetrics(e, test_mean_crossentropy, "test", data.metricsOutPath)
        #write_entropy(e, train_mean_crossentropy, test_mean_crossentropy, data.metricsOutPath)
        
        e += 1
    write_entropy(train_ent, test_ent, data.metricsOutPath)

####### Testing Functions ##########

def test(data):

    obj = object_new(data.D,data.K,data.M)
    matches = 0.0
    
    for i in [0,1]:
        if (i == 1):
            dataset = data.testData 
            outPath = data.testOutPath
            setName = "test"
        else: 
            dataset = data.trainData 
            outPath = data.trainOutPath
            setName = "train"

        N, cols = dataset.shape
        matches = 0.0

        for elem in dataset:
            y = elem[0]
            x = copy.deepcopy(elem[1:])
            obj = forward_propogation(obj, x, y, data.alpha, data.beta)


            prediction = np.argmax(obj.y_hat)
            if (prediction == y): matches += 1

            writePredictions(prediction, outPath)

        error = 1.0 - matches/float(N)
        writeError(error, setName, data.metricsOutPath)

#calculates y, y_hat, J lists of all values


###### Printing FUnctions ##############

def writePredictions(prediction, outPath):
    f = open(outPath, "a")

    f.write("%d\n" % prediction)

    f.close

def writeError(error, dataSetType, outPath):
    f = open(outPath, "a")

    f.write("error(%s): %0.2f\n" % (dataSetType, error))

    f.close

def writeLossToMetrics(numEpochs, loss, dataSetType, outPath):

    f = open(outPath, "a")

    f.write("epoch=%d crossentropy(%s): %0.11f\n" % (numEpochs, dataSetType, loss))

    f.close

def write_entropy(train, test, out):
    f = open(out, "a")

    f.write("\n LOOOK OVER HEARE \n\n")

    for elem in train:
        f.write("%0.11f \n" % (elem))

    f.write("\n test \n\n")

    for elem in test:
            f.write("%0.11f \n" % (elem))

    f.close
######MAIN########
def main(data):
    #BUNCH OF FUNCTIONS TO RUN STUFF
    initialize(data)
    train(data)
    test(data)
if __name__ == "__main__":
    class Struct(object): pass
    data = Struct()

    data.trainData = np.loadtxt(sys.argv[1], delimiter=",")
    data.testData = np.loadtxt(sys.argv[2], delimiter=",")

    data.initFlag = int(sys.argv[8])
    data.learnRate = float(sys.argv[9])
    data.E = int(sys.argv[6]) # numEpochs
    data.D = int(sys.argv[7]) # hidden units
    data.trainOutPath = sys.argv[3]
    data.testOutPath = sys.argv[4]
    data.metricsOutPath = sys.argv[5] 

    f = open(data.trainOutPath, "w")
    f.close

    f = open(data.testOutPath, "w")
    f.close

    f = open(data.metricsOutPath, "w")
    f.close
    main(data)
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
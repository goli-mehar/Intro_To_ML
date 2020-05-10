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





import csv
import sys
import copy

def splitData(dataSet, index, choices):
#REQUIRES: dataSet w/values + index to split on and the choice arguments
#ENSURES: returns two lists: one of choice A, one of choice B

    choiceA = choices[0]
    choiceB = choices[1]

    listA = []
    listB = []

    for element in dataSet:
        if element[index] == choiceA: listA.append(element)
        elif element[index] == choiceB: listB.append(element)

    return listA, listB

def prettyPrint(depth, attributeName, attributeChoice, labelCounts, labelChoices):

    stringOffset = "| "
    stringOffset = "".join([stringOffset]*(depth+1))

    attribute = attributeName + " = " + attributeChoice + " "
    label = "[" + str(labelCounts[0]) + " " + labelChoices[0] + "/" + str(labelCounts[1]) + " " + labelChoices[1] + "]"

    sys.stdout.write(stringOffset + attribute + label + '\n')
##############################################################################################################################
############################################## TREE STRUCTURE FUNCTIONS ######################################################
##############################################################################################################################
class Leaf:
    def __init__(self, index, depth, choices, testMembers, name):
        self.index = index
        self.choices = choices
        self.labelCounts = []
        self.testMembers = testMembers
        self.depth = depth
        self.name = name

def countChoices(dataList, index, choiceA, choiceB):
#REQUIRES: list of strings w/one of two possible choices, 
#          two choices that match w/ possible strings in list
# #ENSURES: returns tuple w/ counts of A, B (countA, countB)

    countA = 0
    countB = 0

    for choice in dataList:
        if choice[index] == choiceA: countA += 1
        else: countB += 1

    return (countA, countB)

def determineGiniImpurity(counts):
#REQUIRES: tuple of counts of choice A and count of choice B
#ENSURES: Returns numeric value of gini impurity, given counts

    countA = float (counts[0])
    countB = float (counts[1])
    total = float (countA + countB)


    giniA = countA/total
    giniA = giniA*giniA

    giniB = countB/total
    giniB = giniB*giniB

    giniImpurity = giniA + giniB
    giniImpurity = float(1.0) - giniImpurity

    return giniImpurity

def newLeaf(index, depth, choices, testMembers, labelChoices, name):
#REQUIRES: index of newLeaf, giniGain, errorRate, choices, labelCounts, testMember
#ENSURES: returns a new leaf node

    leaf = Leaf(index, depth, choices, testMembers, name)
    leaf.labelCounts = countChoices(testMembers, len(testMembers[0]) - 1, labelChoices[0], labelChoices[1])
    
    leaf.giniImpurity = determineGiniImpurity(leaf.labelCounts)

    return leaf


##############################################################################################################################
################################################## INITIALIZE DATA ###########################################################
##############################################################################################################################

def determineChoices(dataList, index):
#REQUIRES: datalist = list of strings w/one of two possible choices
#ENSURES: result is a tuple w/ two possible choices determined

    choiceA = ""
    choiceB = ""

    choiceA = dataList[0][index]
    choiceB = choiceA

    i = 0
    while choiceA == choiceB:
        choiceB = dataList[i][index]
        i += 1

    return (choiceA, choiceB)
          
def initializeData(data):
#INITIALIZE
#ENSURES: Retrieve training data, test data, store in 2D lists
#         Determining list w/ two choices for each attribute, the label
#         running list of indexes left
#         determining number of indexes
    data.trainData = []
    data.testData = []
    data.indexCount = 0
    data.choices = []
    data.splitIndexesLeft = []
    data.titles = []


    
    #Retrieve training data, store in 2D list
    #Each element is a single patient, with their traits of various attributes and final label
    with open(data.trainInPath) as trainInput:

        train = csv.reader(trainInput, delimiter = ' ', quotechar = '|')
        next(train)
        for row in train:
            elems = row[0].split('\t')
            data.trainData.append(elems)

    f = open(data.trainInPath, "r")
    trainData = f.read()
    trainElems = trainData.split('\n')

    data.titles = trainElems[0].split('\t')
    data.indexCount = len(data.titles)
    
    with open(data.testInPath) as testInput:
        test = csv.reader(testInput, delimiter = ' ', quotechar = '|')
        next(test)[0].split('\t')
        for row in test:
            elems = row[0].split('\t')
            data.testData.append(elems)

    #Determining the two choices for each attribute, the label
    #Storing in 2D list first index = attribute index, elements = tuples of choices
    index = 0
    while index < data.indexCount:
        #Making a running list of indexes left
        data.splitIndexesLeft.append(index)

        data.choices.append(determineChoices(data.trainData, index))
        index += 1 

    #removing labels index from possible splitting indexes
    data.splitIndexesLeft.remove(data.indexCount - 1)  




##############################################################################################################################
#################################################### TRAINING ################################################################
##############################################################################################################################


########################################### GINI GAIN #####################################################
def determineGiniGainArguments(dataSet, attributeIndex, attributeChoices, labelIndex, labelChoices):
#REQUIRES: current target data set, splitting index + choices, label index+choices
#ENSURES: returns p(A), gini impurity(group A), p(B), gini impurity(group B)

    

    #choices
    attChoiceA = attributeChoices[0]
    attChoiceB = attributeChoices[1]

    labChoiceA = labelChoices[0]
    labChoiceB = labelChoices[1]

    #size of attribute sub-group group for choice A, corresponding label counts of that group
    sizeA = 0
    groupA_labelA = 0
    groupA_labelB = 0

    #size of attribute sub-group group for choice B, corresponding label counts of that group
    sizeB = 0
    groupB_labelA = 0
    groupB_labelB = 0

    #computing size of attribute sub-group groups for choice A, B
    # and corresponding label counts of each sub-group
    for element in dataSet:
        attribute = element[attributeIndex]
        label = element[labelIndex]
        
        if attribute == attChoiceA:
            sizeA += 1
            if label == labChoiceA: groupA_labelA += 1
            elif label == labChoiceB: groupA_labelB += 1
        elif attribute == attChoiceB:
            sizeB += 1
            if label == labChoiceA: groupB_labelA += 1
            elif label == labChoiceB: groupB_labelB += 1
    totalSize = float(sizeA + sizeB)
    ###Final result variables
    probA = float(sizeA) / totalSize
    giniA = determineGiniImpurity((groupA_labelA, groupA_labelB))
    probB = float(sizeB) / totalSize
    giniB = determineGiniImpurity((groupB_labelA, groupB_labelB))

   
    return probA, giniA, probB, giniB

def determineGiniGain(giniImpurity, dataSet, attributeIndex, attributeChoices, labelIndex, labelChoices):
#REQUIRES: parent giniimpurity, current target data set, splitting index + choices, label index+choices
#ENSURES: returns gini gain of current attribute

    probA, giniA, probB, giniB = determineGiniGainArguments(dataSet, attributeIndex, attributeChoices, labelIndex, labelChoices)

    giniGain = probA*giniA + probB*giniB
    giniGain = giniImpurity - giniGain

    
    return giniGain

########################################### MAJORITY VOTE #####################################################

def majorityVote(counts, labelChoices):
#REUQIRES: counts of a, b and label choices
#ENSURES returns choice which is majority vote of label

    if counts[0] > counts[1]:
        return labelChoices[0]
    elif counts[1] > counts[0]:
        return labelChoices[1]
    else: return max(labelChoices)

########################################### BEST ATTRIBUTE #####################################################
def isValuesEqual(dataSet, index):

    label0 = dataSet[0][index]

    for element in dataSet:
        if label0 != element[index]: 
            return False
    
    return True
    ####WRITE THIS TOMMOROW

def determineBestAttribute(dataSet, availableAttributes, choices, giniImpurity, titles):
#REQUIRES: dataset of current node, available attributes to split + choices
#ENSURES: returns index of best attribute, its gini value

    giniMax = -5.0
    attrMax = 0
    giniCur = -5.0
    attrCur = 0

    labelIndex = len(choices) - 1
    labelChoices = choices[labelIndex]

    for attrCur in availableAttributes:
        if isValuesEqual(dataSet, attrCur) == False: 
            attributeChoices = choices[attrCur]
            giniCur = determineGiniGain(giniImpurity, dataSet, attrCur, attributeChoices, labelIndex, labelChoices)

        
            if giniCur >= giniMax:
                giniMax = giniCur
                attrMax = attrCur
            elif giniCur == giniMax:
                attrMax_title = titles[attrMax]
                attrCur_title = titles[attrCur]
                if max(attrMax_title, attrCur_title) == attrCur_title:
                    attrMax = attrCur
        

    return attrMax, giniMax

def buildNextTreeNode(parentDepth, giniImpurity, dataSet, availableAttributes, choices, maxDepth, titles):
#REQUIRES: given the current parent, available Attributes, choices list, max depth
#ENSURE: returns next tree node, updated attributes list

    depth = parentDepth + 1
    labelIndex = len(choices) - 1
    labelChoices = choices[labelIndex]

    attributesLeft = copy.copy(availableAttributes)

    if (depth + 1 > maxDepth) or (len(availableAttributes) == 0) or isValuesEqual(dataSet, len(dataSet[0]) - 1) or giniImpurity == 0:
        counts = countChoices(dataSet, labelIndex, labelChoices[0], labelChoices[1])
        return majorityVote(counts, labelChoices)

    nextAttribute, giniGain = determineBestAttribute(dataSet, availableAttributes, choices, giniImpurity, titles)
    
    if giniGain <= 0:
        counts = countChoices(dataSet, labelIndex, labelChoices[0], labelChoices[1])
        return majorityVote(counts, labelChoices)

    leafChoices = choices[nextAttribute]
    leafName = titles[nextAttribute]

    leafNew = newLeaf(nextAttribute, depth, leafChoices, dataSet, labelChoices, leafName)

    setA, setB = splitData(dataSet, nextAttribute, leafChoices)

    attributesLeft.remove(nextAttribute)

    #Pretty Print
    aCounts = countChoices(setA, labelIndex, labelChoices[0], labelChoices[1])
    prettyPrint(depth, leafName, leafChoices[0], aCounts, labelChoices)

    #print(attributesLeft, nextAttribute)
    leafNew.left = buildNextTreeNode(leafNew.depth, leafNew.giniImpurity, setA, attributesLeft, choices, maxDepth, titles)

    #Pretty Print
    bCounts = countChoices(setB, labelIndex, labelChoices[0], labelChoices[1])
    prettyPrint(depth, leafName, leafChoices[1], bCounts, labelChoices)

    leafNew.right = buildNextTreeNode(leafNew.depth, leafNew.giniImpurity, setB, attributesLeft, choices, maxDepth, titles)

    return leafNew
    #if tree depth at max, no more attributes

def buildTree(data):
    
    parentDepth = -1
    dataSet = data.trainData
    availableAttributes = data.splitIndexesLeft
    choices = data.choices
    maxDepth = data.maxDepth
    titles = data.titles

    labelChoices = choices[data.indexCount - 1]
    

    labelCount = countChoices(dataSet, data.indexCount - 1, labelChoices[0], labelChoices[1])
    label = "[" + str(labelCount[0]) + " " + labelChoices[0] + "/" + str(labelCount[1]) + " " + labelChoices[1] + "]"
    sys.stdout.write(label + '\n')

    giniImpurity = determineGiniImpurity(labelCount)

    data.tree = buildNextTreeNode(parentDepth, giniImpurity, dataSet, availableAttributes, choices, maxDepth, titles)


def trainAlgorithm(data):
#REQUIRES: data class
#ENSURES: builds a data tree
    buildTree(data)

##############################################################################################################################
#################################################### TESTING #################################################################
##############################################################################################################################

def determineLabelGuess(element, tree):
#REQUIRES: a dataset
#ENSURES: returns prediction

    if type(tree) == str: return tree

    attributeIndex = tree.index
    attributeChoices = tree.choices

    choiceA = attributeChoices[0]
    choiceB = attributeChoices[1]

    elemAttr = element[attributeIndex]
    result = ""

    if elemAttr == choiceA: 
        if type(tree.left) != str: return determineLabelGuess(element, tree.left)
        else: result = tree.left
    elif elemAttr == choiceB: 
        if type(tree.right) != str: return determineLabelGuess(element, tree.right)
        else: result = tree.right

    return result

def determineLabelGuessAccuracy(labelActual, labelGuess):
#REQUIRES: actual label and a guess label
#ENSURES: returns 'Y' or 'N' accordingly
    if labelGuess == labelActual:
        return 'Y'
    else:
        return 'N'

def testDataSet(dataSet, tree, indexCount):
#REQUIRES: data, tree
#ENSURES: returns guesses, matches
    guesses = []
    results = []
    labelIndex = indexCount - 1

    for element in dataSet:
        labelActual = element[labelIndex]
        labelGuess = determineLabelGuess(element, tree)
        guessAccuracy = determineLabelGuessAccuracy(labelActual, labelGuess)
        
        guesses.append(labelGuess)
        results.append(guessAccuracy)
        
    return guesses, results

def determineErrorRate(results):
    accuracy = 0
    for element in results:
        if element != 'Y':
            accuracy += 1
    
    return float(accuracy) / float(len(results))

def testData(data):
    data.trainGuesses, data.trainResults = testDataSet(data.trainData, data.tree, data.indexCount)
    data.testGuesses, data.testResults = testDataSet(data.testData, data.tree, data.indexCount)
    
    data.trainError = determineErrorRate(data.trainResults)
    data.testError = determineErrorRate(data.testResults)

def testAlgorithm(data):
    testData(data)


##############################################################################################################################
#################################################### DATA OUTPUT #############################################################
##############################################################################################################################

def dataOut(path, predictions):
    #REQUIRES: path to output file, list of predictions
    #ENSURES writes predictions to path file

    f = open(path, "w")
    
    for element in predictions:
        f.write(element + '\n')
        
    f.close

def metricsOut(path, trainError, testError):
#REQUIRES: path to output file, train, test errors
#ENSURES writes error to path file

    f = open(path, "a")
    
    f.write("error(train): %0.6f\n" %trainError)
    f.write("error(test): %0.6f\n" %testError )
    
    f.close


def printResults(data):
#output

    dataOut(data.trainOutPath, data.trainGuesses)
    dataOut(data.testOutPath, data.testGuesses)
    metricsOut(data.metricsOutPath, data.trainError, data.testError)

##############################################################################################################################
######################################################### MAIN ###############################################################
##############################################################################################################################

def main(data):
    initializeData(data)
    trainAlgorithm(data)
    testAlgorithm(data)
    printResults(data)

if __name__ == "__main__":
    class Struct(object): pass
    data = Struct()

    data.trainInPath = sys.argv[1]
    data.testInPath = sys.argv[2]
    data.maxDepth = int(sys.argv[3])
    data.trainOutPath = sys.argv[4]
    data.testOutPath = sys.argv[5]
    data.metricsOutPath = sys.argv[6]

    main(data)
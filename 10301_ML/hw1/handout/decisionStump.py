#<train input> <test input> <split index> <train out> <test out> <metrics out>
import csv
import sys
#parse arguments given to system

#Retrieving list of of datasets of training input and test input
#returns a 'data' structure w:
#training, test lists where elemets are [attribute, element] (attribute from splitting index)
#initializes what the possible labels are
def initializeData(data):
    data.trainData = []
    data.testData = []
    data.attributeGroupA = []
    data.attributeGroupB = []
    
    #Used python documentation: https://docs.python.org/3/library/csv.html?highlight=csv
    #for accessing data in files
    with open(data.trainInput) as trainInput:
        trainInputReader = csv.reader(trainInput, delimiter = ' ', quotechar = '|')
        next(trainInputReader)
        for row in trainInputReader:
            elems = row[0].split('\t')
            attribute = elems[int(data.attributeIndex)]
            diagnosis = elems[(len(elems) - 1)]
            traits = [attribute, diagnosis]
            data.trainData.append(traits)
        
    with open(data.testInput) as testInput:
        testInputReader = csv.reader(testInput, delimiter = ' ', quotechar = '|')
        next(testInputReader)
        for row in testInputReader:
            elems = row[0].split('\t')
            attribute = elems[int(data.attributeIndex)]
            diagnosis = elems[(len(elems) - 1)]
            traits = [attribute, diagnosis]
            data.testData.append(traits)
    
    #determine attribute values
    data.attributeA = data.trainData[1][0]
    data.attributeB = data.attributeA
    
    i = 2
    while data.attributeA == data.attributeB:
        data.attributeB = data.trainData[i][0]
        i += 1
    
    #dtermine diagnosis values
    data.diagnosisA = data.testData[1][1]
    data.diagnosisB = data.diagnosisA
    
    i = 2
    while data.diagnosisA == data.diagnosisB:
        data.diagnosisB = data.testData[i][1]
        i += 1

##Training

##splits data based on chosen attribute
#creates two groups for each attribute set
def splitData(data):
    for patient in data.trainData:
        attribute = patient[0]
        diagnosis = patient[1]
        
        if attribute == data.attributeA:
            data.attributeGroupA.append(diagnosis)
        else:
            data.attributeGroupB.append(diagnosis)
                
#takes in a list of diagnosis, computes which diagnosis is the majority  
def majorityVote(diagnosisList, diagnosisA, diagnosisB):
    diagnosisACount = 0
    diagnosisBCount = 0
    
    for diagnosis in diagnosisList:
        if diagnosis == diagnosisA: diagnosisACount += 1
        elif diagnosis == diagnosisB: diagnosisBCount += 1
        
    if diagnosisACount > diagnosisBCount: return diagnosisA
    else: return diagnosisB
        
def performMajVote(data):
    data.majVoteGroupA = majorityVote(data.attributeGroupA, data.diagnosisA, data.diagnosisB)
    data.majVoteGroupB = majorityVote(data.attributeGroupB, data.diagnosisA, data.diagnosisB)
    
def trainAlgorithm(data):
    splitData(data)
    performMajVote(data)
    
##Testing
    
def determineGuessDiagnosis(data, attribute):
    if attribute == data.attributeA:
        return data.majVoteGroupA
    else:
        return data.majVoteGroupB
        
def determineDiagnosisAccuracy(actual, guess):
    if guess == actual:
        return 'Y'
    else:
        return 'N'
    
def testData(data, sourceData):
    results = []
    for patient in sourceData:
        attribute = patient[0]
        actualDiagnosis = patient[1]
        guessDiagnosis = determineGuessDiagnosis(data, attribute)
        diagnosisAccuracy = determineDiagnosisAccuracy(actualDiagnosis, guessDiagnosis)
        
        valuesList = [guessDiagnosis, diagnosisAccuracy]
        results.append(valuesList)
        
    return results
        
def determineErrorRate(results):
    accuracy = 0
    for element in results:
        diagnosisAccuracy = element[1]
        if diagnosisAccuracy != 'Y':
            accuracy += 1
    
    return float(accuracy) / float(len(results))
    
def testAlgorithm(data):
    
    data.trainResults = testData(data, data.trainData)
    data.testResults = testData(data, data.testData)
    
    data.trainError = determineErrorRate(data.trainResults)
    data.testError = determineErrorRate(data.testResults)
    
##Printing results

def printGuesses(results, filePath):
    f = open(filePath, "w")
    
    for element in results:
        guessDiagnosis = element[0]
        f.write(guessDiagnosis + '\n')
        
    f.close
    
def printError(trainError, testError, filePath):
    f = open(filePath, "w")
    
    f.write("error(train): %0.6f\n" %trainError)
    f.write("error(test): %0.6f\n" %testError )
    
    f.close
    
def printResults(data):
    printGuesses(data.trainResults, data.trainOutput)
    printGuesses(data.testResults, data.testOutput)
    printError(data.trainError, data.testError, data.metricsOutput)
    
### Overall File Running

def main(data):
    initializeData(data)
    trainAlgorithm(data)
    testAlgorithm(data)
    printResults(data)

if __name__ == "__main__":
    class Struct(object): pass
    data = Struct()
    data.trainInput = sys.argv[1]
    data.testInput = sys.argv[2]
    data.attributeIndex = sys.argv[3]
    data.trainOutput = sys.argv[4]
    data.testOutput = sys.argv[5]
    data.metricsOutput = sys.argv[6]

    main(data)
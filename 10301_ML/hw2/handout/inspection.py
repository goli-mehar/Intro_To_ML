import csv
import sys

############## Support Functions ###################################################


def determineChoices(dataList):
#REQUIRES: datalist = list of strings w/one of two possible choices
#ENSURES: result is a tuple w/ two possible choices determined

    choiceA = ""
    choiceB = ""

    choiceA = dataList[0]
    choiceB = choiceA

    i = 1
    while choiceA == choiceB:
        choiceB = dataList[i]
        i += 1

    return choiceA, choiceB

def countChoices(dataList, choiceA, choiceB):
#REQUIRES: list of strings w/one of two possible choices, 
#          two choices that match w/ possible strings in list
#ENSURES: returns tuple w/ counts of A, B (countA, countB)

    countA = 0
    countB = 0

    for choice in dataList:
        if choice == choiceA: countA += 1
        else: countB += 1

    return countA, countB

def majorityVote(dataList, choiceA, choiceB, countA, countB):
# REQUIRES: determines majority vote of data set
# ENSURES: returns choice that is the majority vote

    if countA > countB: return choiceA
    else: return choiceB


############## INITIALIZE DATA ###########################################################

def initializeData(data):
    #access data files
    # dtermine labels
    data.dataList = []
    data.labelA = ""
    data.labelB = ""
    data.countA = 0
    data.countB = 0

    with open(data.inputPath) as inp:
        reader = csv.reader(inp, delimiter = ' ', quotechar = '|')
        next(reader)
        for row in reader:
            elems = row[0].split('\t')
            label = elems[(len(elems) - 1)]
            data.dataList.append(label)

    data.labelA, data.labelB = determineChoices(data.dataList)
    data.countA, data.countB = countChoices(data.dataList, data.labelA, data.labelB)


########### CALCULATE GINI GAIN, ERROR RATE #################################################

def determineGiniGain(data):
# REQUIRES: data class
# ENSURES: sets a new variable called gini gain

    data.giniGain = 0

    countA = float (data.countA)
    countB = float (data.countB)
    total = countA + countB

    data.giniGain = 1.0 - (countA/total)*(countA/total) - (countB/total)*(countB/total)

def determineErrorRate(data):
# REQUIRES: data class
# ENSURES: sets a new variable called error rate

    data.errorRate = 0
    
    countA = float (data.countA)
    countB = float (data.countB)
    total = countA + countB

    majVote = majorityVote(data.dataList, data.labelA, data.labelB, data.countA, data.countB)

    if majVote == data.labelA: data.errorRate = countB/total
    else: data.errorRate = countA/total


############## PRINT FINAL OUTPUT TO FILE ################################################

def reportData(data):
# REQUIRES: data class
# ENSURES: writes error, gini gain to given path

    path = data.outputPath
    error = data.errorRate
    gini = data.giniGain

    f = open(path, "w")
        
    f.write("gini_impurity: %0.4f\n" %gini)
    f.write("error: %0.4f\n" %error )
            
    f.close


######################### MAIN ########################################################

def main(data):
    initializeData(data)
    determineGiniGain(data)
    determineErrorRate(data)
    reportData(data)


if __name__ == "__main__":
    class Struct(object): pass
    data = Struct()
    data.inputPath = sys.argv[1]
    data.outputPath = sys.argv[2]

    main(data)


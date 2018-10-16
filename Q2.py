#https://www.kaggle.com/uciml/pima-indians-diabetes-database
import csv
import random
import math
import operator
import numpy as np
import pandas as pd 


def loadDataset(train_file,test_file, trainingSet=[], testSet=[]):

    train = pd.read_csv(train_file)
    train = train[:1000]
    train = train.as_matrix()
    test = pd.read_csv(test_file)
    test = test[:15]
    test = test.as_matrix()

    for i in range(len(train)):
        trainingSet.append(train[i])

    for j in range(len(test)):
        testSet.append(test[j])



def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(1,length):
        distance += pow((float(instance1[x]) - float(instance2[x])), 2)#here is a type transaction
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}

    for x in range(len(neighbors)):
        response = neighbors[x][0]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    print (classVotes)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    # TRUE POSITIVE & TRUE NEGATIVE & FALSE POSITIVE & FALSE NEGATIVE
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for x in range(len(testSet)):
        if testSet[x][-1] == 1.0 and predictions[x] == 1.0:
            tp += 1
        if testSet[x][-1] == 0.0 and predictions[x] == 0.0:
            tn += 1
        if testSet[x][-1] == 0.0 and predictions[x] == 1.0:
            fp += 1
        if testSet[x][-1] == 1.0 and predictions[x] == 0.0:
            fn += 1
    print('# OF TRUE POSITIVE: ' + str(tp))
    print('# OF TRUE NEGATIVE: ' + str(tn))
    print('# OF FALSE POSITIVE: ' + str(fp))
    print('# OF FALSE NEGATIVE: ' + str(fn))
    correct = int(tp) + int(tn)
    total = correct + int(fp) + int(fn)
    return float(correct)/float(total)

def getCon_matrix(testSet,predictions):
    con_ma = np.zeros(shape=[10,10])
    for x in range(len(testSet)):
        con_ma[testSet[x][0]][predictions[x]] += 1
    print (con_ma)




def main():
    # prepare data
    trainingSet = []
    testSet = []

    
    loadDataset('./data/MNIST/train.csv','./data/MNIST/test.csv', trainingSet, testSet)

    # formalized the np array output
    np.set_printoptions(precision=3, threshold=10, linewidth=90, suppress=True)
    print('Train set: ')
    #print(trainingSet)
    print('Test set: ')
    #print(testSet)
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))

    # generate predictions and calculate the accuracy
    predictions = []
    k = 11 # was 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][0]))

    getCon_matrix(testSet,predictions)
    #print(type(testSet[0][0])) #was trying to modify the data type so that it can calculate the accuracy
    #print(type(predictions[0]))
    #accuracy = getAccuracy(testSet, predictions)
    #print('Accuracy: ' + str(accuracy*100) + '%')


main()

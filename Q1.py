#https://www.kaggle.com/uciml/pima-indians-diabetes-database
import csv
import random
import math
import time
import operator
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.model_selection import train_test_split

def load_data(file,split,trainingSet=[], testSet=[]):
    dataset = pd.read_csv(file)
    columns = dataset.columns
    dataset[columns.drop('Outcome')] = StandardScaler().fit_transform(dataset[columns.drop('Outcome')])
    #dataset[columns.drop('Outcome')] = MinMaxScaler().fit_transform(dataset[columns.drop('Outcome')])
    y_label = dataset['Outcome']
    train, test = train_test_split(dataset,train_size = split)#,stratify=y_label)#,random_state = 45)
    training = train.as_matrix()
    testing = test.as_matrix()

    for x in range(len(training)):
        trainingSet.append((training[x]))

    for y in range(len(testing)):
        testSet.append(testing[y])     



def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)

        # Remove headers
        dataset = np.array(list(lines)[1:][:], dtype=np.float64)
        np.set_printoptions(threshold=np.nan,suppress=True)
        #print('==========ORIGINAL DATASET==========\n')
        #print(dataset)
        # standardize all values except output
        dataset[:, :-1] = StandardScaler().fit_transform(dataset[:, :-1])
        #dataset[:, :-1] = MinMaxScaler().fit_transform(dataset[:, :-1])
        np.set_printoptions(threshold=np.nan,suppress=True, linewidth=10000)
        #print('\n==========STANDARDIZED DATASET==========\n')
        #print(dataset)

        for x in range(0, len(dataset)):
            for y in range(len(dataset[0])):#
                dataset[x][y] = (dataset[x][y])
            
            if random.random() < split: # shuffle the data before split
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
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
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
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


def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.8 # 0.73 is the best
    #loadDataset('./data/diabetes.csv', split, trainingSet, testSet)
    load_data('./data/diabetes.csv', split,trainingSet, testSet)
    # formalized the np array output
    np.set_printoptions(precision=3, threshold=10, linewidth=90, suppress=True)
    #print('Train set: ')
    #print(trainingSet)
    #print('Test set: ')
    #print(testSet)
    #print 'Train set: ' + repr(len(trainingSet))
    #print 'Test set: ' + repr(len(testSet))

    # generate predictions and calculate the accuracy
    ts1 = time.time()
    predictions = []
    k = 11 # was 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    #print(type(testSet[0][0])) #was trying to modify the data type so that it can calculate the accuracy
    #print(type(predictions[0]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + str(accuracy*100) + '%')
    print ('The whole process time is %s',time.time()-ts1)

main()

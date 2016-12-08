import math
import operator
from utilities import *


def dist(instance1, instance2):
    d = 0
    for i in range(len(instance1)):
        d += pow(instance1[i][0] - instance2[i][0], 2) + pow(instance1[i][1] - instance2[i][1], 2) \
             + pow(instance1[i][2] - instance2[i][2], 2)
    distance = math.sqrt(d)
    return distance


def getNeighbors(testInstance, k, trainingSet):
    distances = []

    for vals in trainingSet.values():
        d = dist(testInstance[1], vals[1])
        distances.append((vals, d))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    voting = {}
    for x in range(len(neighbors)):
        response = neighbors[x][0]
        if response in voting:
            voting[response] += 1
        else:
            voting[response] = 1
    sortedNeighbors = sorted(voting.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedNeighbors[0][0]


def getAccuracy(testSet, predictions):

    correct = 0
    count = 0
    iterator = 0
    for vals in testSet.values():
        if vals[0] == predictions[iterator]:
            iterator += 1
            correct += 1
            count += 1

    return (correct/float(len(testSet))) * 100.0

if __name__ == '__main__':
    trainingSet=[]
    testSet=[]

    testDict = convertToRGB('test-data.txt')
    trainingSet = convertToRGB('train-data.txt')

    k = 6
    predictions = []
    for val in testDict.values():
        neighbors = getNeighbors(val, k, trainingSet)
        result = getResponse(neighbors)
        predictions.append(result)

    accuracy = getAccuracy(testDict, predictions)
    print 'Accuracy ' + str(accuracy)

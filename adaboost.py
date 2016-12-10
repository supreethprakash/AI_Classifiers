from utilities import *
import random


def ensemle(dataset, numOfStumps, degree):
    counter1 = 0
    ensemble0 = []
    x, y = 2, 4
    for i in range(numOfStumps):
        weights = [1] * len(dataset)
        bestSplit = []
        bestCounter = 0
        bestError = 0
        counter1 += 1
        counter = 0
        error = 0
        counter2 = 0
        for j in dataset.values():
            if weights[counter2] == 1:
                z = 1
                counter2 += 1
                if j[1][x] > j[1][y]:
                    z = degree
                if int(j[0]) == z:
                    counter += 1
                    weights[counter2] = 0
                else:
                    error += 1
                if counter > bestCounter:
                    bestCounter = counter
                    bestSplit = [x,y]
                    bestError = error
        alpha = float(1 - (float(bestError)/len(dataset)))/len(dataset)
        alpha = alpha * 10000
        x += 2
        y += 3
        l = []
        l.append(alpha)
        l.append(bestSplit)
        ensemble0.append(l)
    return ensemble0


if __name__ == '__main__':
    trainingSet=[]
    testSet=[]

    testDict = convertToRGB('test-data.txt')
    trainingSet = convertToRGB('train-data.txt')


    e0 = ensemle(trainingSet, 30,0)
    e1 = ensemle(trainingSet, 30,90)
    e2 = ensemle(trainingSet, 30,180)
    e3 = ensemle(trainingSet, 30,270)
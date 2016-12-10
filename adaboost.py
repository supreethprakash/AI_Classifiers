from utilities import *
import random
import operator


def calculateBestSplit(datatset, degree, weight, ensemble):
    dataLength = len(datatset[0][1])
    x, y = random.sample(set(list(range(dataLength))), 2)
    leastError = 40000
    if x != y:
        error = 0
        counter = 0
        tempWeights = weight
        for records in datatset.values():
            tempDegree = 1
            if weight[counter] == 1:
                if records[1][x] > records[1][y]:
                    tempDegree = degree
                if tempDegree == records[0]:
                    tempWeights[counter] = 0
                else:
                    error += 1
            counter += 1
        if error < leastError:
            bestAtt1 = x
            bestAtt2 = y
            if (bestAtt1, bestAtt2) not in ensemble:
                weight = tempWeights
                alpha = float(1 - (float(error) / len(datatset))) / len(datatset)
                alpha = alpha * 100000
                l = []
                l.append(alpha)
                l.append((x, y))
                ensemble.append(l)
        else:
            calculateBestSplit(dataset,degree,weight,ensemble)
    return ensemble



def ensemle(dataset, numOfStumps, degree):

    ensemble = []
    #weight = [1] * len(dataset)
    for i in range(numOfStumps):
        ensemble = calculateBestSplit(dataset,degree,weight, ensemble)
    return ensemble


def testing(e0,e1,e2,e3,dataset):
    correctness = 0
    for record in dataset.values():
        counterDict = {'counter0':0, 'counter1':0, 'counter2':0, 'counter3':0}
        for i in range(len(e0)):
            val = e0[i][0]
            att = e0[i][1]
            x = att[0]
            y = att[1]
            if record[1][x] > record[1][y]:
                counterDict['counter0'] += 1
            else:
                counterDict['counter0'] -= 1
        for i in range(len(e1)):
            val = e1[i][0]
            att = e1[i][1]
            x = att[0]
            y = att[1]
            if record[1][x] > record[1][y]:
                counterDict['counter1'] += 1
            else:
                counterDict['counter1'] -= 1
        for i in range(len(e2)):
            val = e2[i][0]
            att = e2[i][1]
            x = att[0]
            y = att[1]
            if record[1][x] > record[1][y]:
                counterDict['counter2'] += 1
            else:
                counterDict['counter2'] -= 1
        for i in range(len(e3)):
            val = e3[i][0]
            att = e3[i][1]
            x = att[0]
            y = att[1]
            if record[1][x] > record[1][y]:
                counterDict['counter3'] += 1
            else:
                counterDict['counter3'] -= 1

        maximumScore = sorted(counterDict.iteritems(), key=operator.itemgetter(1), reverse=True)
        vote = maximumScore[0][0]
        if vote == 'counter0':
            c = 0
        elif vote == 'counter1':
            c = 90
        elif vote == 'counter2':
            c =180
        else:
            c =270
        if c == int(record[0]):
            correctness += 1

    return correctness

if __name__ == '__main__':
    trainingSet = []
    testSet = []

    testDict = convertToRGB('test-data.txt')
    trainingSet = convertToRGB('train-data.txt')
    weight = [1] * len(trainingSet)
    e0 = ensemle(trainingSet, 50, 0)
    weight = [1] * len(trainingSet)
    e1 = ensemle(trainingSet, 50, 90)
    weight = [1] * len(trainingSet)
    e2 = ensemle(trainingSet, 50, 180)
    weight = [1] * len(trainingSet)
    e3 = ensemle(trainingSet, 50, 270)

    c = testing(e0,e1,e2,e3,testDict)

    print e0
    print e1
    print e2
    print e3

    print c
    print len(testDict)
    print (float(c)/len(testDict)) * 100
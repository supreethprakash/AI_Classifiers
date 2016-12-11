from utilities import *
import random
import operator


def calculateBestSplit(datatset, degree, weight, ensemble, success, error):
    dataLength = len(datatset[0][1])
    x, y = random.sample(set(list(range(dataLength))), 2)
    maxWeight = max(weight)
    su = max(weight)
    er = min(weight)
    if x != y:
        counter = 0
        tempSuccess = 0
        tempError = 0
        for records in datatset.values():
            tempDegree = 1
            if weight[counter] == maxWeight:
                if records[1][x] > records[1][y]:
                    tempDegree = degree
                if tempDegree == records[0]:
                    tempSuccess += 1
                else:
                    tempError += 1
            counter += 1
        if tempSuccess >= success and tempError <= error:
            counterNew = 0
            success = tempSuccess
            error = tempError
            errorWeight = float(tempError)/len(datatset)
            errorWeight = (er * errorWeight)/(1-errorWeight)
            for records in datatset.values():
                tempDegree = 1
                if weight[counterNew] == maxWeight:
                    if records[1][x] > records[1][y]:
                        tempDegree = degree
                    if tempDegree == records[0]:
                        weight[counterNew] = errorWeight
                counterNew += 1
            bestAtt1 = x
            bestAtt2 = y
            if (bestAtt1, bestAtt2) not in ensemble:
                alpha = float(1 - (float(error) / len(datatset))) / len(datatset)
                alpha = alpha * 100000
                l = []
                l.append(alpha)
                l.append((x, y))
                ensemble.append(l)
    return ensemble



def ensemle(dataset, numOfStumps, degree, weight, success, error):

    ensemble = []
    for i in range(numOfStumps):
        ensemble = calculateBestSplit(dataset,degree,weight,ensemble, success, error)
        sumOfWeights = sum(weight)
        weight = [float(i)/sumOfWeights for i in weight]
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
                counterDict['counter0'] += val * 1
            else:
                counterDict['counter0'] += val * -1
        for i in range(len(e1)):
            val = e1[i][0]
            att = e1[i][1]
            x = att[0]
            y = att[1]
            if record[1][x] > record[1][y]:
                counterDict['counter1'] += val * 1
            else:
                counterDict['counter1'] += val * -1
        for i in range(len(e2)):
            val = e2[i][0]
            att = e2[i][1]
            x = att[0]
            y = att[1]
            if record[1][x] > record[1][y]:
                counterDict['counter2'] += val * 1
            else:
                counterDict['counter2'] += val * -1
        for i in range(len(e3)):
            val = e3[i][0]
            att = e3[i][1]
            x = att[0]
            y = att[1]
            if record[1][x] > record[1][y]:
                counterDict['counter3'] += val * 1
            else:
                counterDict['counter3'] += val * -1

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
    orientation = [0,90,180,270]
    points = float(1)/len(trainingSet)
    for i in orientation:
        weight = [points] * len(trainingSet)
        success = 0
        error = len(trainingSet)
        e = ensemle(trainingSet,50,i, weight, success, error)
        if i == 0:
            e0 = e
        elif i == 90:
            e1 = e
        elif i == 180:
            e2 = e
        else:
            e3 =e

    c = testing(e0,e1,e2,e3,testDict)

    print e0
    print e1
    print e2
    print e3

    print c
    print len(testDict)
    print (float(c)/len(testDict)) * 100
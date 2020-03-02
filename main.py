from math import sqrt
from random import randrange

def eDistance(x1, x2):
    dist = 0.0
    for i in range(len(x1)-1):
        dist += (x1[i]-x2[i])**2
    return sqrt(dist)

def readFile(fileName):
    data = list()
    file = open(fileName)
    for line in file:
        lineData = line.split()
        tempLine = list()
        for column in lineData[:-1]:
            tempLine.append(float(column))
        data.append(tempLine)
    return data

def kNeighbors(trainingData, row, k):
    distances = list()
    for trainRow in trainingData:
        dist = eDistance(row, trainRow)
        distances.append((trainRow, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def predict(trainingData, row, k):
    neighbors = kNeighbors(trainingData, row, k)
    output = [row[-1] for row in neighbors]
    prediction = max(set(output), key=output.count)
    return prediction

def kNearest(trainingData, testData, k):
    results = list()
    for row in testData:
        prediction = predict(trainingData, row, k)
        results.append(prediction)

    return results

def meanSquareError(actual, prediction):
    sumOfDiff = 0
    for i in range(len(actual)):
        sumOfDiff += (actual[i][-1]-prediction[i])**2
    return sumOfDiff / len(actual)

def splitData(data, k):
    split = list()
    copy = list(data)
    foldSize = int(len(data) / k)
    for j in range(k):
        fold = list()
        while len(fold) < foldSize:
            i = randrange(len(copy))
            fold.append(copy.pop(i))
        split.append(fold)
    return split

def testKValue(dataset, k):
    folds = splitData(dataset, 5)
    errors = list()
    for fold in folds:
        training = list(folds)
        training.remove(fold)
        training = sum(training, [])
        testing = list()
        for row in fold:
            copy = list(row)
            testing.append(copy)
        knn = kNearest(training, testing, k)
        errors.append(meanSquareError(testing, knn))
    return sum(errors) / 5


trainingData = readFile("./data/u1-base.base")
testData = readFile("./data/u1-test.test")



for i in range(5):
    k = i + 1
    print(k, "nearest neighbors:")
    avgError = testKValue(testData, k)
    print("Average error:", avgError, "\n")

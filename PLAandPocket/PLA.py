import numpy as np



def loadDataInfo(file):
    data = []
    label = []
    lineNum = 0
    with open(file) as f:
        line = f.readline()
        while line:
            lineArray = line.split() # type list
            for i in range(len(lineArray) - 1):
                data.append(float(lineArray[i]))
            label.append(int(lineArray[4]))
            line = f.readline()
            lineNum += 1
    dataArray = np.array(data)
    dataArray = dataArray.reshape(lineNum,4)

    labelArray = np.array(label)
    return dataArray,labelArray


def TestPacket(TestData,TestLabel,weight):
    errorNum = 0
    correctNum = 0
    TestData = np.insert(TestData, 4, [1], axis=1)  # insert x0 column
    for i in range(len(TestData)):
        if np.dot(weight, TestData[i]) * TestLabel[i] <= 0:
            errorNum += 1
        else:
            correctNum += 1
    print ('Test result : correct num is %d, error num is %d, error rate is %f'
              % (correctNum,errorNum,errorNum / (correctNum + errorNum)))


def sign(x):
    if x <= 0:
        return -1
    return 1

def errorRate(data,label,w):
    error = 0
    for i in range(len(data)):
        if sign(np.dot(w, data[i])) != label[i]:
            error += 1
    return error / len(data)

def TrainPacket(data,label):
    """
        pocket algorithm, this can classify two types.
    :return:
        weight about classification
    """
    data = np.insert(data, 0, [1], axis=1) # insert x0 column

    batchNum = 0
    trainSum = 0

    for loop in range(2000):
        w = np.array([.0, .0, .0, .0,.0])
        #data, label = Shuffle2ArrayMeanwhile(data, label, loop)
        batchNum += 1
        trainNum = 0
        seed = 0

        error_w = errorRate(data,label,w)

        IsFinished = False

        while (1):
            seed = seed + 1
            np.random.seed(seed)
            randNum = np.random.rand(1)
            correctNum = 0
            for loopi in range(len(data)):
                index = (int(randNum * len(data)) + loopi) % len(data)
                if sign(np.dot(w, data[index])) !=  label[index]:
                    w = w + data[index] * label[index]
                    error_tryW = errorRate(data, label, w)
                    if (error_tryW < error_w):
                        error_w = error_tryW
                        w_pocket = w
                        correctNum += 1

            if (correctNum == 50):
                break
            """
                    error_new = 0
                    for loopj in range(len(data)):
                        if np.dot(new_w, data[loopj]) * label[loopj] <= 0:
                            error_new += 1
                    print('i=',loopi,'w = ', w, ',new_w = ', new_w,'error_w=',error_w, 'error_new=',error_new)
                    if (error_new < error_w):
                        error_w = error_new
                        w = new_w
                        print ('the last errorNum is %d, the current errorNum is %d' % (error_w, error_new))
                        trainNum += 1
                        if (trainNum > 50):
                            IsFinished = True
                            break
                        break

            if (True == IsFinished):
                break
        trainSum += trainNum
        print('The %d batchs, train num is %d' % (batchNum, trainNum))
        """
    print('The average train num is %f in 2000 batch data' % (trainSum / batchNum))
    print('The w is', w)
    return w

def Shuffle1Array(data):
    return np.random.shuffle(data)

def Shuffle2ArrayMeanwhile(data,label,seed):
    """
       shuffle the data and label meanwhile.
    :return:
       the shuffled data array and shuffled label array
    """
    randomData = []
    randomLabel = []

    for i in range(len(data)):
        np.random.seed(seed)
        randNum = np.random.rand(1)
        randomIndex = int(randNum[0] * len(data))
        randomData.append(data[randomIndex])
        data = np.delete(data,randomIndex,0)
        randomLabel.append(label[randomIndex])
        label = np.delete(label,randomIndex,0)

    data = np.array(randomData)
    label = np.array(randomLabel)
    return data,label

def PLA(data,label):
    """
       PLA algorithm, this can classify two types.
    :return:
       classification standard
    """
    data = np.insert(data,4, [1],  axis=1)
    batchNum = 0
    trainSum = 0
    for loop in range(2000):
        w = np.array([.0, .0, .0, .0, .0])
        data,label = Shuffle2ArrayMeanwhile(data,label,loop)
        batchNum += 1
        trainNum = 0
        while (1):
            errorNum = 0
            correctNum = 0
            for i in range(len(data)):
                if  np.dot(w, data[i]) * label[i] <= 0:
                    w = w + data[i] * label[i]
                    trainNum += 1
                    correctNum = 0
                    errorNum += 1
                else:
                    correctNum += 1
            if 0 == errorNum:
                break
        trainSum += trainNum
        print ('The %d batchs, train num is %d'% (batchNum,trainNum))
    print ('The average train num is %f in 2000 batch data' % (trainSum / batchNum))

def Test_Shuffle2ArrayMeanwhile():
    data = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
    label = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
    for loop in range(10):
        (data,label) = Shuffle2ArrayMeanwhile(data,label,loop)
        print (data,label)



if  __name__ == "__main__":
    """
    PlaFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\PLAandPocket\\data\\plaData.txt'
    data,label = loadDataInfo(PlaFilePath)
    PLA(data,label)
    """

    PlaFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\PLAandPocket\\data\\packetTrainData.txt'
    data,label = loadDataInfo(PlaFilePath)
    weight = TrainPacket(data,label)

    PlaFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\PLAandPocket\\data\\packetTestData.txt'
    data,label = loadDataInfo(PlaFilePath)
    TestPacket(data,label,weight)

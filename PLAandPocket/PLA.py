import numpy as np

def loadDataInfo():
    file = 'G:\\林轩田教程\\MachineLearningFoundations\\PLAandPocket\\data\\data.txt'
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


def pocket():
    pass

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
    for i in range(2000):
        w = np.array([.0, .0, .0, .0, .0])
        data,label = Shuffle2ArrayMeanwhile(data,label,i)
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
    data,label = loadDataInfo()
    #Test_Shuffle2ArrayMeanwhile()
    PLA(data,label)

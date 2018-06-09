import numpy as np
from matplotlib.pyplot import *
from scipy import *

def LoadDataInfo(file):
    data = []
    label = []
    lineNum = 0
    with open(file) as f:
        line = f.readline()
        while line:
            lineArray = line.split()
            for i in range(len(lineArray) - 1):
                data.append(float(lineArray[i]))
            label.append(int(lineArray[4]))
            line = f.readline()
            lineNum += 1
    dataArray = np.array(data)
    dataArray = dataArray.reshape(lineNum,4)

    labelArray = np.array(label)
    return dataArray,labelArray

def sign(x):
    if x <= 0:
        return -1
    return 1

def TestPacket(TestData,TestLabel,weight):
    errorCount = 0
    sumCount = 0
    TestData = np.insert(TestData, 4, [1], axis=1)  # insert x0 column
    for i in range(len(TestData)):
        sumCount += 1
        if sign(np.dot(weight, TestData[i])) != TestLabel[i]:
            errorCount += 1
    print ('Test result : sum num is %d, error rate is %f' % (sumCount,errorCount / sumCount))
    return errorCount / sumCount

def errorRateWithW(data, label, W):
    error = 0
    for i in range(len(data)):
        if sign(np.dot(W, data[i])) != label[i]:
            error += 1
    return error / len(data)

def GetRandomStartLoc(arrayLen,randomSeed):
    np.random.seed(randomSeed)
    return np.random.random_integers(0,arrayLen - 1)

def TrainPacketInQ18(data, label, updateCount=100):
    """
        pocket algorithm, this can classify two types.
    :return:
        weight about classification
    """
    data = np.insert(data, 4, [1], axis=1) # insert x0 column
    loopCount = 0

    data, label = Shuffle2ArrayMeanwhile(data, label, loop)
    loopCount += 1
    seed = 0
    w = np.array([.0, .0, .0, .0, .0])
    error_w = errorRateWithW(data,label,w)
    for loopUpdate in range(updateCount):
        seed = seed + 1
        for loopi in range(len(data)):
            index = GetRandomStartLoc(len(data),seed * len(data) + loopi)
            if sign(np.dot(w, data[index])) !=  label[index]:
                w = w + data[index] * label[index]
                error_tryW = errorRateWithW(data, label, w)
                if (error_tryW < error_w):
                    error_w = error_tryW
                    w_pocket = w
    return w_pocket


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
    data = np.insert(data,4, [1],  axis=1)
    trainNum = 0
    UpdateWSum = 0
    for loop in range(2000):
        w = np.array([.0, .0, .0, .0, .0])
        data,label = Shuffle2ArrayMeanwhile(data,label,loop)
        trainNum += 1
        updateWCountIn1Train = 0
        while (1):
            errorNum = 0
            for i in range(len(data)):
                if  sign(np.dot(w, data[i])) !=  label[i]:
                    w = w + data[i] * label[i]
                    updateWCountIn1Train += 1
                    errorNum += 1
            if 0 == errorNum:
                break
        UpdateWSum += updateWCountIn1Train
        print ('The train num is %d , update w count which in 1 train is %d'% (trainNum,updateWCountIn1Train))
    print ('The average update w num is %f in 2000 batch data' % (UpdateWSum / trainNum))

def Test_Shuffle2ArrayMeanwhile():
    data = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
    label = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
    for loop in range(10):
        (data,label) = Shuffle2ArrayMeanwhile(data,label,loop)
        print (data,label)


def TrainPacketInQ19(data,label,updateCount=50):
    """
        pocket algorithm, this can classify two types.
    :return:
        weight about classification
    """
    data = np.insert(data, 4, [1], axis=1) # insert x0 column
    loopCount = 0

    data, label = Shuffle2ArrayMeanwhile(data, label, loop)
    loopCount += 1
    seed = 0
    w = np.array([.0, .0, .0, .0, .0])
    error_w = errorRateWithW(data,label,w)
    updateWCount = 0
    while(1):
        seed = seed + 1
        for loopi in range(len(data)):
            index = GetRandomStartLoc(len(data),seed * len(data) + loopi)
            if sign(np.dot(w, data[index])) !=  label[index]:
                w = w + data[index] * label[index]
                error_tryW = errorRateWithW(data, label, w)
                updateWCount += 1
                error_w = error_tryW
        if (updateWCount > 50):
            break

    print('The train num is ',loopCount,',errorRate is',error_w,', w_pocket is ',w)
    return w

def DisplayPCA(data,label):
    """
        这里 x 的维数较高，不方便直接进行可视化。
        但之前我有上过吴恩达的《Machine Learning》课程，
        里面介绍了一种叫PCA(主成分分析）的方法，可以用于数据降维，从而实现可视化
    :return:
    """
    # firstly, normalize the data

    x = data[:, 0:4]  # take the input out
    x_norm = x.copy()  # make sure the base is None

    x_mean = x_norm.mean(axis=0)
    x_std = x_norm.std(axis=0)
    x_norm = (x_norm - x_mean) / x_std

    # calculate the covariance matrix
    x_cov = x_norm.T.dot(x_norm) / 400

    # do SVD
    U, S, V = np.linalg.svd(x_cov)
    UReduce = U[:, 0:2]  # take the first two dimensions
    z = x_norm.dot(UReduce)

    z1 = z[where(label[:] == 1)]
    z2 = z[where(label[:] == -1)]

    fig = figure()
    ax = subplot(111)
    ax.plot(z1[:, 0], z1[:, 1], '*', label='$y = 1$')
    ax.plot(z2[:, 0], z2[:, 1], 'r*', label='$y = -1$')
    title('Visualization of Dataset')
    ax.legend(loc='upper left', fontsize='small')
    fig.show()

if  __name__ == "__main__":
    """
    # Q16
    PlaFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\PLAandPocket\\data\\plaData.txt'
    data,label = LoadDataInfo(PlaFilePath)
    PLA(data,label)
    """

    PlaFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\PLAandPocket\\data\\packetTrainData.txt'
    data, label = LoadDataInfo(PlaFilePath)
    DisplayPCA(data,label)

    errorRateList = []
    for loop in range(20):
        PlaFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\PLAandPocket\\data\\packetTrainData.txt'
        data,label = LoadDataInfo(PlaFilePath)
        weight = TrainPacketInQ18(data, label)
        #weight = TrainPacketInQ19(data, label)

        PlaFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\PLAandPocket\\data\\packetTestData.txt'
        data,label = LoadDataInfo(PlaFilePath)
        error_w = TestPacket(data,label,weight)
        errorRateList.append(error_w)
    errorRateArray = np.array(errorRateList)
    print('The final average error rate is',round(np.average(errorRateArray),4))
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

def PLA(data,label):
    w = np.array([.0,.0,.0,.0,.0])
    trainNum = 0
    correctNum = 0

    data = np.insert(data,4, [1],  axis=1)
    while (1):
        for i in range(len(data)):
            if  np.dot(w, data[i]) * label[i] <= 0:
                w = w + data[i] * label[i]
                trainNum += 1
                correctNum = 0
            else:
                correctNum += 1

        print (trainNum,correctNum)


if  __name__ == "__main__":
    data,label = loadDataInfo()
    PLA(data,label)

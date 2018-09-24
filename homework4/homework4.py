import numpy as np

def DoQuestion13_15():
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
                label.append(int(lineArray[2]))
                line = f.readline()
                lineNum += 1
        dataArray = np.array(data)
        dataArray = dataArray.reshape(lineNum, 2)

        labelArray = np.array(label)
        return dataArray, labelArray

    def sign(a):
        if a >= 0:
            return 1
        elif a < 0:
            return -1

    TrainFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\homework4\\data\\question13_TRAIN.txt'
    TestFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\homework4\\data\\question13_TEST.txt'
    lamda =pow(10,0)
    x_train,y_train = LoadDataInfo(TrainFilePath)
    x_train = np.insert(x_train,2,1,axis=1)
    factor1 = np.linalg.inv(np.add(np.dot(lamda,np.eye(3)),np.dot(np.transpose(x_train),x_train)))
    factor2 = np.dot(np.transpose(x_train),y_train)
    print(factor1.shape,factor2.shape)
    w_reg = np.dot(factor1 , factor2)
    print (w_reg)
    print (w_reg.shape)

    error = 0
    for loop in range(x_train.shape[0]):
        x = np.zeros(3)
        x[0],x[1],x[2] = x_train[loop][0],x_train[loop][1],1
        y_predict = np.dot(np.transpose(w_reg),x)
        if sign(y_predict) != y_train[loop]:
            error += 1

    print ('E_in:',error/x_train.shape[0])

    x_test, y_test = LoadDataInfo(TestFilePath)
    error = 0
    for loop in range(x_test.shape[0]):
        x = np.zeros(3)
        x[0],x[1],x[2] = x_test[loop][0],x_test[loop][1],1
        y_predict = np.dot(np.transpose(w_reg),x)
        if sign(y_predict) != y_test[loop]:
            error += 1

    print ('E_out:',error/x_test.shape[0])


def  DoQuestion16():
    lamda = [pow(10, 0),pow(10, -2),pow(10, -4),pow(10, -6),pow(10, -8)]
    for loop in range(len(lamda)):
        print ('lamda:',lamda[loop])
        CalcEinWithLamda(lamda[loop])

def CalcEinWithLamda(lamda):
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
                label.append(int(lineArray[2]))
                line = f.readline()
                lineNum += 1
        dataArray = np.array(data)
        dataArray = dataArray.reshape(lineNum, 2)

        labelArray = np.array(label)
        return dataArray, labelArray

    def sign(a):
        if a >= 0:
            return 1
        elif a < 0:
            return -1

    TrainFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\homework4\\data\\question13_TRAIN.txt'
    TestFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\homework4\\data\\question13_TEST.txt'
    x_train,y_train = LoadDataInfo(TrainFilePath)
    x_train = np.insert(x_train,2,1,axis=1)

    x_split = np.split(x_train,[120])
    y_split = np.split(y_train,[120])


    x_train = x_split[0]
    y_train = y_split[0]
    factor1 = np.linalg.inv(np.add(np.dot(lamda,np.eye(3)),np.dot(np.transpose(x_train),x_train)))
    factor2 = np.dot(np.transpose(x_train),y_train)
    w_reg = np.dot(factor1 , factor2)
    print (w_reg)

    error = 0
    for loop in range(x_train.shape[0]):
        x = np.zeros(3)
        x[0],x[1],x[2] = x_train[loop][0],x_train[loop][1],1
        y_predict = np.dot(np.transpose(w_reg),x)
        if sign(y_predict) != y_train[loop]:
            error += 1

    print ('E_in:',error/x_train.shape[0])

    x_val = x_split[1]
    y_val = y_split[1]
    error = 0
    for loop in range(x_val.shape[0]):
        x = np.zeros(3)
        x[0],x[1],x[2] = x_val[loop][0],x_val[loop][1],1
        y_predict = np.dot(np.transpose(w_reg),x)
        if sign(y_predict) != y_val[loop]:
            error += 1

    print ('E_val:',error/y_val.shape[0])


    x_test, y_test = LoadDataInfo(TestFilePath)
    error = 0
    for loop in range(x_test.shape[0]):
        x = np.zeros(3)
        x[0],x[1],x[2] = x_test[loop][0],x_test[loop][1],1
        y_predict = np.dot(np.transpose(w_reg),x)
        if sign(y_predict) != y_test[loop]:
            error += 1

    print ('E_out:',error/x_test.shape[0])


def DoQuestion19_20():
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
                label.append(int(lineArray[2]))
                line = f.readline()
                lineNum += 1
        dataArray = np.array(data)
        dataArray = dataArray.reshape(lineNum, 2)

        labelArray = np.array(label)
        return dataArray, labelArray

    def sign(a):
        if a >= 0:
            return 1
        elif a < 0:
            return -1

    TrainFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\homework4\\data\\question13_TRAIN.txt'
    TestFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\homework4\\data\\question13_TEST.txt'
    x_train,y_train = LoadDataInfo(TrainFilePath)
    x_train = np.insert(x_train,2,1,axis=1)

    x_split = np.split(x_train,[40,80,120,160])
    y_split = np.split(y_train,[40,80,120,160])

    lamda = pow(10,-8)
    E_min = 1

    errorCnt = 0
    count    = 0
    for loopTest in range(5):
        startFlag = 0
        for loopTrain in range(5):
            if loopTest == loopTrain:
                continue
            if 0 == startFlag:
                startFlag = 1
                x_trainData = x_split[loopTrain]
                y_trainData = y_split[loopTrain]
            else:
                x_trainData = np.concatenate((x_trainData, x_split[loopTrain]), axis=0)
                y_trainData = np.concatenate((y_trainData, y_split[loopTrain]), axis=0)
        #print (x_trainData.shape,y_trainData.shape)

        factor1 = np.linalg.inv(np.add(np.dot(lamda,np.eye(3)),np.dot(np.transpose(x_trainData),x_trainData)))
        factor2 = np.dot(np.transpose(x_trainData),y_trainData)
        w_reg = np.dot(factor1 , factor2)
        #print (w_reg)


        x_val = x_split[loopTest]
        y_val = y_split[loopTest]

        for loop in range(x_val.shape[0]):
            x = np.zeros(3)
            x[0],x[1],x[2] = x_val[loop][0],x_val[loop][1],1
            y_predict = np.dot(np.transpose(w_reg),x)
            #print (y_predict,y_val[loop])
            count += 1
            if sign(y_predict) != y_val[loop]:
                errorCnt += 1
    print('lamda:',lamda,'error:', errorCnt/count)

    # 先根据之前的交叉验证选择最优的lamda，然后将使用该lamda对所有的训练数据进行训练得到 W_reg
    x_train, y_train = LoadDataInfo(TrainFilePath)
    x_train = np.insert(x_train, 2, 1, axis=1)
    factor1 = np.linalg.inv(np.add(np.dot(lamda, np.eye(3)), np.dot(np.transpose(x_train), x_train)))
    factor2 = np.dot(np.transpose(x_train), y_train)
    w_regAllData = np.dot(factor1, factor2)

    error = 0
    for loop in range(x_train.shape[0]):
        x = np.zeros(3)
        x[0],x[1],x[2] = x_train[loop][0],x_train[loop][1],1
        y_predict = np.dot(np.transpose(w_regAllData),x)
        if sign(y_predict) != y_train[loop]:
            error += 1

    print ('E_in:',error/x_train.shape[0])

    x_test, y_test = LoadDataInfo(TestFilePath)
    error = 0
    for loop in range(x_test.shape[0]):
        x = np.zeros(3)
        x[0],x[1],x[2] = x_test[loop][0],x_test[loop][1],1
        y_predict = np.dot(np.transpose(w_regAllData),x)
        if sign(y_predict) != y_test[loop]:
            error += 1

    print ('E_out:',error/x_test.shape[0])


# 面向对象的手法看起来挺精妙的，学习了，后续逐渐更换风格为面向对象的
def fun():
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
                label.append(int(lineArray[2]))
                line = f.readline()
                lineNum += 1
        dataArray = np.array(data)
        dataArray = dataArray.reshape(lineNum, 2)

        labelArray = np.array(label)
        return dataArray, labelArray

    def sign(a):
        if a >= 0:
            return 1
        elif a < 0:
            return -1

    class LinearRegressionReg:

        def __init__(self):
            self._dimension = 0

        def fit(self, X, Y, lamb):
            self._dimension = len(X[0])
            self._w = np.zeros((self._dimension, 1))
            self._lamb = lamb
            self._w = np.linalg.inv(np.dot(X.T, X) + lamb * np.eye(self._dimension)).dot(X.T).dot(Y)

        def predict(self, X):
            result = np.dot(X, self._w)
            return np.array([(1 if _ >= 0 else -1) for _ in result]).reshape(len(X), 1)

        def score(self, X, Y):
            Y_predict = self.predict(X)
            return np.sum(Y_predict != Y) / (len(Y) * 1.0)

        def get_w(self):
            return self._w

        def print_val(self):
            print("w: ", self._w)

    TrainFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\homework4\\data\\question13_TRAIN.txt'
    TestFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\homework4\\data\\question13_TEST.txt'
    lamda =1
    x_train,y_train = LoadDataInfo(TrainFilePath)
    ### Error in
    lr = LinearRegressionReg()
    lr.fit(x_train, y_train, 11.26)
    Ein = lr.score(x_train, y_train)

    lr.print_val()
    print ("Ein : ", Ein)

    error = 0
    for loop in range(x_train.shape[0]):
        x = np.zeros(2)
        x[0],x[1] = x_train[loop][0],x_train[loop][1]
        y_predict = np.dot(np.transpose(lr.get_w()),x)
        if sign(y_predict) != y_train[loop]:
            error += 1

    print ('E_in:',error/x_train.shape[0])

if __name__ == "__main__":
    #DoQuestion16()
    #DoQuestion13_15()
    DoQuestion19_20()
    #fun()
import numpy as np
import matplotlib.pyplot as mp
import scipy
import math
import sympy
from decimal import Decimal


def DoQuestion6And7():
    loopNum = 5
    array = np.zeros((2))

    def GetPartialD(array):
        # E(u,v) = exp(u) + exp(2v) + exp(uv) + u^2 - 2uv + 2 * (v^2) - 3*u - 2*v
        u = array[0]
        v = array[1]
        uOfPartialD = np.exp(u) + v * np.exp(u * v) + 2 * u - 2 * v - 3
        vOfPartialD = 2 * np.exp(2 * v) + u * np.exp(u * v) -2 * u + 4 * v - 2
        return np.array((uOfPartialD,vOfPartialD))

    eta = 0.01
    print ('Question6 Answer is:' , GetPartialD(array))
    for i in range(loopNum):
        array = array - eta * GetPartialD(array)

    def CalcError(array):
        u = array[0]
        v = array[1]
        return np.exp(u) + np.exp(2*v) + np.exp(u*v) + np.power(u,2) - 2*u*v + 2 * (np.power(v,2)) - 3*u - 2*v

    print ('Question7 Answer is:' , CalcError(array))

def DoQuestion10():
    # E(u,v) = exp(u) + exp(2v) + exp(uv) + u^2 - 2uv + 2 * (v^2) - 3*u - 2*v
    # x_n+1 = x_n - (f'(x)) / (f''(x))
    loop = 5

    def GetHessianMatrix(u,v):
        dEuv_u =  np.exp(u) + v * np.exp(u * v) + 2 * u - 2 * v - 3
        dEuv_u_v = v * u * np.exp(u * v) - 2
        dEuv_v = 2 * np.exp(2 * v) + u * np.exp(u * v) -2 * u + 4 * v - 2
        ddEuv_u =  np.exp(u) + v * v * np.exp(u * v) + 2
        ddEuv_v = 4 * np.exp(2 * v) + u * u * np.exp(u * v)  + 4
        # hessianMatrix = np.zeros((2, 2))
        hessianMatrix = np.array([[ddEuv_u, dEuv_u_v],[dEuv_u_v, ddEuv_v]])
        print ('hessianMatrix shape:',hessianMatrix.shape)
        return hessianMatrix

    def GetJacobianMatrix(u,v):
        dEuv_u = np.exp(u) + v * np.exp(u * v) + 2 * u - 2 * v - 3
        dEuv_v = 2 * np.exp(2 * v) + u * np.exp(u * v) - 2 * u + 4 * v - 2
        JacobianMatrix = np.array([dEuv_u,dEuv_v]) # 这里处理有些奇怪，定于为 2*1的矩阵会报异常
        print('JacobianMatrix shape:',JacobianMatrix.shape)
        return JacobianMatrix

    array = np.zeros((2))
    for loop in range(5):
        array = array - np.dot(np.linalg.inv(GetHessianMatrix(array[0],array[1])),GetJacobianMatrix(array[0],array[1]))
        print (array)

    def CalcError(array):
        u = array[0]
        v = array[1]
        return np.exp(u) + np.exp(2*v) + np.exp(u*v) + np.power(u,2) - 2*u*v + 2 * (np.power(v,2)) - 3*u - 2*v

    print ('Question10 Answer is:' , CalcError(array))

def DoQuestion13():
    def GenerateUniform():
        array = np.random.uniform(-1,1,(1000,2))    # shape =  [1000 * 3]
        array = np.insert(array, 0, 1, axis=1)
        print (array.shape)
        return array

    def sign(a):
        if a >= 0:
            return 1
        elif a < 0:
            return -1

    def GenerateLabelArray(array):
        label = np.zeros(1000)
        for loop in range(1000):
            label[loop] = sign(array[loop][1]*array[loop][1] + array[loop][2]*array[loop][2] - 0.6)
        return label

    def GenerateNoise(label):
        randomIndex = np.random.random_integers(0,999,100)
        print ('randomIndex:',randomIndex,'len:',randomIndex.size)
        for loop in range(randomIndex.size):
            label[randomIndex[loop]] = -1 * label[randomIndex[loop]]
        return label

    trainNum = 1000
    errorSum = 0
    for trainCnt in range(trainNum):
        array = GenerateUniform()
        oriLabel = GenerateLabelArray(array)
        label = GenerateNoise(oriLabel)
        # w_lin = inverse(X_T * X) * X_T * Y
        w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(array),array)) ,np.transpose(array)) ,label)
        errCount = 0
        for loop in range(1000):
            x = np.array([1,array[loop][1],array[loop][2]])
            if sign(np.dot(w,x)) != oriLabel[loop]:
                errCount += 1
                errorSum += 1
        print (errCount/1000)
    print ('E_in',errorSum / trainNum)

def DoQuestion14and15():
    def GenerateUniform():
        array = np.random.uniform(-1,1,(1000,2))    # {1, x1, x2, x1*x2, x1*x1, x2*x2}
        array = np.insert(array, 0, 1, axis=1)
        return array

    def sign(a):
        if a >= 0:
            return 1
        elif a < 0:
            return -1

    def GenerateLabelArray(array):
        label = np.zeros(1000)
        for loop in range(1000):
            label[loop] = sign(array[loop][1]*array[loop][1] + array[loop][2]*array[loop][2] - 0.6)
        return label

    def GenerateNoise(label):
        randomIndex = np.random.random_integers(0,999,100)
        for loop in range(randomIndex.size):
            label[randomIndex[loop]] = -1 * label[randomIndex[loop]]
        return label

    trainNum = 1000
    minEinCnt  = 1000
    w_best = np.zeros(6)

    e_outOfErr = 0
    e_outOfCnt = 0
    e_inOfErr = 0
    e_inOfCnt = 0
    for trainCnt in range(trainNum):
        array = GenerateUniform()
        y_ori = GenerateLabelArray(array)
        y = GenerateNoise(y_ori)
        # w_lin = inverse(X_T * X) * X_T * Y
        try:

            z = np.array([array[:,1],array[:,2],\
                          array[:,1] * array[:,2],\
                          array[:,1]*array[:,1],array[:,2]*array[:,2]])  # {1, x1, x2, x1*x2, x1*x1, x2*x2}
            z = np.transpose(z)
            z = np.insert(z,0,1,axis=1)
            w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(z), z)), np.transpose(z)), y)

            for loop in range(1000):
                e_inOfCnt += 1
                if sign(np.dot(z[loop,:],np.transpose(w))) != y[loop]:
                    e_inOfErr += 1

            if (minEinCnt > e_inOfErr):
                minEinCnt = e_inOfErr
                w_best = w

            x_TEST = GenerateUniform()
            y_ori_TEST = GenerateLabelArray(x_TEST)
            y_TEST = GenerateNoise(y_ori_TEST)
            z_TEST = np.array([x_TEST[:,1],x_TEST[:,2], \
                               x_TEST[:,1] * x_TEST[:,2], \
                               x_TEST[:,1] * x_TEST[:,1],x_TEST[:,2]*x_TEST[:,2]])  # {1, x1, x2, x1*x2, x1*x1, x2*x2}
            z_TEST = np.transpose(z_TEST)
            z_TEST = np.insert(z_TEST,0,1,axis=1)

            for loop in range(1000):
                e_outOfCnt += 1
                if sign(np.dot(z_TEST[loop,:],np.transpose(w))) != y_TEST[loop]:
                    e_outOfErr += 1

        except Exception as e:
            print(e.args)
        print ('best_w',w_best)

    print('E_in', e_inOfErr / e_inOfCnt)
    print ('E_out',e_outOfErr/e_outOfCnt)


def DoQuestion18And19():
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
                label.append(int(lineArray[20]))
                line = f.readline()
                lineNum += 1
        dataArray = np.array(data)
        dataArray = dataArray.reshape(lineNum, 20)

        labelArray = np.array(label)
        return dataArray, labelArray

    def sign(a):
        if a >= 0:
            return 1
        elif a < 0:
            return -1


    def sigmoid(x):
        return 1 / (1 + np.exp(np.multiply(-1,x)))

    def GetGradient(x,y,w):
        gradient = 0
        for loop in range(x.shape[0]):
            gradient += sigmoid(np.multiply(-1 * y[loop] , np.dot(w, x[loop]))) \
                      * (np.multiply(-1 * y[loop] , x[loop]))
        gradient = gradient / x.shape[0]

        return gradient

    def FindW(ita,T,x,y,w):
        for loop in range(T):
            gradient = GetGradient(x,y,w)
            w = w - np.multiply(ita,gradient)
        print('gradient:',GetGradient(x,y,w))
        return w


    TrainFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\homework3\\data\\question18_Train.txt'
    TestFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\homework3\\data\\question18_Test.txt'
    data, label = LoadDataInfo(TrainFilePath)
    print (data.shape,label.shape)

    iTa = 0.01   # 随机梯度步长
    T   = 2000   # 可能是随机梯度的循环次数 iterations
    w = np.zeros(20)

    w = FindW(iTa,T,data,label,w)

    data, label = LoadDataInfo(TestFilePath)
    e_outError = 0
    e_outCount = 0
    for loop in range(data.shape[0]):
        e_outCount += 1
        if sign(np.dot(w,data[loop])) != label[loop]:  # 不太清楚这里为什么不能用 sigmoid(wx) 而是直接用 sign(wx)
            e_outError+= 1
    print ('E_out:',e_outError/e_outCount)

def DoQuestion20():
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
                label.append(int(lineArray[20]))
                line = f.readline()
                lineNum += 1
        dataArray = np.array(data)
        dataArray = dataArray.reshape(lineNum, 20)

        labelArray = np.array(label)
        return dataArray, labelArray

    def sign(a):
        if a >= 0:
            return 1
        elif a < 0:
            return -1


    def sigmoid(x):
        return 1 / (1 + np.exp(np.multiply(-1,x)))

    def GetGradient(x,y,w):
        gradient = sigmoid(np.multiply(-1 * y , np.dot(w, x))) \
                  * (np.multiply(-1 * y , x))

        return gradient

    def FindW(ita,T,x,y,w):
        gradient = GetGradient(x,y,w)
        w = w - np.multiply(ita,gradient)
        print('gradient:',gradient)
        return w


    TrainFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\homework3\\data\\question18_Train.txt'
    TestFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\homework3\\data\\question18_Test.txt'
    data, label = LoadDataInfo(TrainFilePath)
    print (data.shape,label.shape)

    iTa = 0.001   # 随机梯度步长
    T   = 2000   # 可能是随机梯度的循环次数 iterations
    w = np.zeros(20)

    for loop in range(T):
        w = FindW(iTa,T,data[loop % data.shape[0]],label[loop % data.shape[0]],w)

    data, label = LoadDataInfo(TestFilePath)
    e_outError = 0
    e_outCount = 0
    for loop in range(data.shape[0]):
        e_outCount += 1
        if sign(np.dot(w,data[loop])) != label[loop]:  # 不太清楚这里为什么不能用 sigmoid(wx) 而是直接用 sign(wx)
            e_outError+= 1
    print ('E_out:',e_outError/e_outCount)

if  __name__ == "__main__":
    DoQuestion20()
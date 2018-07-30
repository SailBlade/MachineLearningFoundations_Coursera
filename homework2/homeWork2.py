import numpy as np
import matplotlib.pyplot as mp
import scipy
import math
import sympy
from decimal import Decimal

def DoQuestion3():
    fig = mp.figure()
    ax = mp.subplot(111)
    elpsilon = 0.05
    d_vc = 10
    x = []
    y = []
    loop = 0
    var = 1000
    for N in range(10000000):
        if scipy.mod(N,100) == 5:
            var = var * 3
            loop += 1
            value = 4 *((2*N) ** d_vc) * math.exp(-(elpsilon**2)*N/8)
            if (1 == loop):
                lastValue = value
            x.append(N)
            y.append(value)

            print (value - 0.05,lastValue - 0.05)
            if  scipy.sign(value - 0.05) != scipy.sign(lastValue - 0.05):
                print (scipy.sign(value - 0.05),scipy.sign(lastValue - 0.05))
                break
            lastValue = value


    z1 = np.array(x)
    z2 = np.array(y)
    ax.plot(z1[:], z2[:], '*', label='$y = Value$')
    #top = 10**10
    #ax.set_ylim(0,top)

    mp.title('Visualization of Dataset')
    ax.legend(loc='upper left', fontsize='small')
    fig.show()

def DoQuestion4And5():
    delta = 0.05
    d_vc = 50
    N = 10000

    # Original VC bound:
    elpsilon1 = math.sqrt((8 / N)*math.log((4 * ((2 * N)**d_vc))/ delta))

    # Rademacher Penalty Bound
    elpsilon2 = math.sqrt(2 * sympy.log((2 * N * ((N)**d_vc)))/ N) + math.sqrt((2/N)* sympy.log(1/delta)) + 1 / N

    # Parrondo and Van den Broek
    elpsilon3 = sympy.Symbol("elpsilon3")
    lnFactor =  round(math.log((6 * ((2 * N)**d_vc))/ delta),2)
    elpsilon3 = sympy.solve(sympy.sqrt((2 * elpsilon3 + lnFactor)/ N) - elpsilon3,elpsilon3)

    # Devroye
    elpsilon4 = sympy.Symbol("elpsilon4")
    exp1 =  4 * (Decimal(N ** 2) ** d_vc)/ Decimal(delta)
    lnFactor = exp1.ln()  # math.log返回inf
    elpsilon4 = sympy.solve(sympy.sqrt((1 / (2 * N)) * (4 * elpsilon4 * (1 + elpsilon4) + lnFactor)) - elpsilon4,elpsilon4)

    # Variant VC bound
    elpsilon5 = sympy.Symbol("elpsilon5")
    elpsilon5 = math.sqrt((16 / N) * math.log((2 * ((N) ** d_vc))/sympy.sqrt(delta)))

    print ('Original VC bound',elpsilon1,
           '\nRademacher Penalty Bound: ',elpsilon2,
           '\nParrondo and Van den Broek: ',elpsilon3[0],
           '\nDevroye: ',elpsilon4[0],
           '\nelpsilon5: ',elpsilon5)

def sign(x):
    if x <= 0:
        return -1
    return 1


def DoQuestion17_method1():
    LoopNum = 5000
    dataLen = 20
    noiseProbality = 0.2

    Sum_Ein = 0
    Sum_Eout = 0
    for loop in range(LoopNum):
        E_in,E_out = SearchEinInPositiveAndNegativeRays(dataLen,noiseProbality)
        Sum_Ein += E_in
        Sum_Eout += E_out
    print ('Average E_in:',Sum_Ein / LoopNum,',Average E_out:',Sum_Eout / LoopNum)


def SearchEinInPositiveAndNegativeRays(dataLen,noiseProbality):
    data = np.random.uniform(low=-1.0, high=1.0, size=dataLen)
    data = sorted(data)  # 这个排序属于隐藏条件，不排序E_in只有0.3，排序后有0.15
    s_x = [sign(element) for element in data]
    noiseLocation = np.random.random_integers(low=0, high=dataLen - 1, size=int(dataLen * noiseProbality))

    for theta in range(int(dataLen * noiseProbality)):  # 随机生成四个位置设置为噪点，将符号位取反
        s_x[noiseLocation[theta]] = -1 * s_x[noiseLocation[theta]]

    Ein = len(s_x)
    LocationOfMinE_in = 0
    signOfMinE_in = -1
    leftSign = -1
    rightSign = 1
    for symbor in range(2):  # 因为是 positive and negative rays 所以需要尝试两次
        leftSign = -1 * leftSign
        rightSign = -1 * rightSign
        for theta in range(len(s_x) + 1):
            leftErrorNum = 0
            rightErrorNum = 0
            leftNum = 0
            rightNum = 0
            for i in range(theta):
                if i >= len(s_x):
                    continue
                leftNum += 1
                if leftSign != s_x[i]:
                    leftErrorNum += 1
            for i in range(len(s_x) + 1 - theta):
                if i + theta >= len(s_x):
                    continue
                rightNum += 1
                if rightSign != s_x[i + theta]:
                    rightErrorNum += 1
            sumError = leftErrorNum + rightErrorNum
            if (sumError < Ein):
                Ein = sumError
                LocationOfMinE_in = theta
                signOfMinE_in = leftSign

    if LocationOfMinE_in + 1 < len(s_x):
        bestTheta = (s_x[LocationOfMinE_in] + s_x[LocationOfMinE_in + 1])/2
    else:
        bestTheta = 1
    E_out = 0.5 + 0.3 * signOfMinE_in * (np.abs(bestTheta) - 1)
    print ('ErrorRate: ',Ein/len(s_x),LocationOfMinE_in,',E_out: ',E_out)
    return Ein/len(s_x), E_out

def signArray(x):#自定义符号函数，只返回-1，+1
    ret=np.ones(x.shape)
    for i,each in enumerate(x):
        if each<0: ret[i]=-1
    return ret

def getTheta(x):  # 由输入的x生成假设空间的所有theta的序列
    n = len(x)
    l1 = sorted(x) #为什么要排序？
    theta = np.zeros(n)
    for i in range(n - 1):
        theta[i] = (l1[i] + l1[i + 1]) / 2
    theta[-1] = 1
    return theta


def DoQuestion17_method2():
    data_size = 3
    expes = 5000
    E_in = 0
    E_out = 0
    for i in range(expes):
        x = np.random.uniform(-1, 1, data_size)
        noise_rate = 0.2
        # 生成[-0.2,0.8]范围内的随机数组，取sign()即变为有20%的-1的随机数组
        noise = signArray(np.random.uniform(size=data_size) - noise_rate)
        y = signArray(x) * noise  # 为y加上20%的噪声
        theta = getTheta(x)
        print(x)
        print(theta)
        e_in = np.zeros((2, data_size))  # 对每个theta求出一个error_in,第一行是s=1，第2行是s=-1.
        for i in range(len(theta)):
            a1 = y * signArray(x - theta[i])
            print ('a1:',a1,',np.sum(a1):',np.sum(a1),',np.sum(-a1):',np.sum(-a1))
            e_in[0][i] = (data_size - np.sum(a1)) / (2 * data_size)  # 数组只有-1和+1，可直接计算出-1所占比例
            e_in[1][i] = (data_size - np.sum(-a1)) / (2 * data_size)
            print ('E_in[0]:',e_in[0][i],',e_in[1][i]:',e_in[1][i])
        s = 0
        theta_best = 0
        min0, min1 = np.min(e_in[0]), np.min(e_in[1])
        if min0 < min1:
            s = 1
            theta_best = theta[np.argmin(e_in[0])]
        else:
            s = -1
            theta_best = theta[np.argmin(e_in[1])]
        e_out = 0.5 + 0.3 * s * (np.abs(theta_best) - 1)
        E_in += np.min(e_in)
        E_out += np.min(e_out)
    ave_in = E_in / expes
    ave_out = E_out / expes
    print(ave_in,ave_out)


def TestSympy():
    a = sympy.Symbol('a')
    y = sympy.solve(sympy.sqrt(2*a)-4,a)
    print (y)


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
            label.append(int(lineArray[9]))
            line = f.readline()
            lineNum += 1
    dataArray = np.array(data)
    dataArray = dataArray.reshape(lineNum,9)

    labelArray = np.array(label)
    return dataArray,labelArray


def DoDecisionStump(data, label):
    data_size = len(data)
    theta = getTheta(data)
    e_in = np.zeros((2, data_size))
    for i in range(len(theta)):
        a1 = label * signArray(data - theta[i])
        e_in[0][i] = (data_size - np.sum(a1)) / (2 * data_size)
        e_in[1][i] = (data_size - np.sum(-a1)) / (2 * data_size)

    s = 0
    min0, min1 = np.min(e_in[0]), np.min(e_in[1])
    if min0 < min1:
        s = 1
        theta_best = theta[np.argmin(e_in[0])]
    else:
        s = -1
        theta_best = theta[np.argmin(e_in[1])]
    E_in = np.min(np.min(e_in))
    return s, theta_best, E_in

def getData_i(X_train,i):#获取第d维数据
    return np.reshape(X_train[:,i],len(X_train))#从ndarray二维数组转为array一维数组


def DoQuestion19():
    PlaFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\homework2\\data\\decisionTrumpTrainData.txt'
    data, label = LoadDataInfo(PlaFilePath)
    e_in = np.zeros(9)
    s = np.zeros(9)
    theta = np.zeros(9)

    for i in range(9):
        s[i], theta[i], e_in[i] = DoDecisionStump(getData_i(data,i),label)

    E_in = np.min(e_in)
    dimension = np.argmin(e_in)
    theta_best = theta[dimension]
    s_best = s[dimension]

    PlaFilePath = 'G:\\林轩田教程\\MachineLearningFoundations\\homework2\\data\\decisionTrumpTestData.txt'
    data, label = LoadDataInfo(PlaFilePath)
    test_len = len(label)
    X_i = getData_i(data, dimension)
    q = label * s_best * signArray(X_i - theta_best)
    E_out = (test_len - np.sum(q)) / (2 * test_len)
    print(E_in, E_out)

if  __name__ == "__main__":
    # DoQuestion3()
    #DoQuestion4And5()

    #DoQuestion17_method1()
    DoQuestion17_method2() # 网上提供的另外一种思路
    #DoQuestion19()
    pass

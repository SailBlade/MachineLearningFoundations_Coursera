import numpy as np
import cvxopt as cvx
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import math

#from homework5.homework5 import LinearSVM

global YLoc
YLoc = 2

global weightLoc
weightLoc = 3

global featureNum
featureNum = 2

def DoAdaptiveBoost():

    class AdaptiveBoost():
        def __init__(self):
            self.YLoc = 2    # Y值的位置
            self._trainLen = 20
            self._trainNum = 20

        def __sign__(self,a):
            if a >= 0.0:
                return 1
            elif a < 0.0:
                return -1

        def __generateTrainData__(self):
            X1 = np.random.uniform(-1.0,1.0,self._trainLen)
            X2 = np.random.uniform(-1.0,1.0,self._trainLen)
            randomY  = np.random.uniform(-1.0,1.0,self._trainLen)
            Y = []
            for loop in range(self._trainLen):
                Y.append(self.__sign__(randomY[loop]))

            Y = np.array(Y)

            #X1 = np.array([-0.24477302,0.19475279,0.25548862,-0.54018758,0.30140678,-0.34477604,0.13941788,0.16259198,0.26696617,-0.75998417])
            #X2 = np.array([ 0.51412259,0.28742792,0.4249421, 0.36054774,-0.60541032,0.3538119,-0.043742,0.52602336,0.76907062,0.99818954])
            #Y  = np.array([-1,-1,-1,1,1,-1,1,-1,1,1])
            self.X1 = X1
            self.X2 = X2
            self.Y  = Y
            self.DataSet = np.column_stack((X1, X2, Y))
            self.trainSet = self.DataSet

        def __genTrainSetWithBootstrapping__(self,seed):
            randomArray = np.around(np.random.uniform(0, self._trainLen - 1, self._trainLen), decimals=0)

            trainSet = np.zeros((1, 3))

            '''
            randomArray = np.zeros(1)
            for loop in range(self._trainLen):
                randomArray = np.append(randomArray,loop)

            randomArray = np.delete(randomArray,0,0)
            
            for loop in range(randomArray.size):
                if seed == randomArray.size - 1:
                    randomArray[randomArray.size - 1] = randomArray[0]
                else:
                    randomArray[seed] = randomArray[seed + 1]
            '''

            for loop in range(randomArray.size):
                arrayIndex = int(randomArray[loop])
                element = self.DataSet[arrayIndex]
                element = np.reshape(element,(1,3))
                trainSet = np.concatenate((trainSet, element), axis=0)

            trainSet = np.delete(trainSet, 0, 0)
            self.trainSet = trainSet[np.lexsort(trainSet[:, ::-1].T)]  # 令人窒息的按第一列排序的操作

        def __genTrainInterVal__(self,featureIdx):
            intervalList = []
            self.trainSet = self.trainSet[np.lexsort(self.trainSet[:, ::-1].T)]  # 令人窒息的按第一列排序的操作
            for loop in range(self.trainSet.shape[0]):
                if ((loop + 1) == self.trainSet.shape[0]):
                    break
                intervalList.append((self.trainSet[loop][featureIdx] + self.trainSet[loop + 1][featureIdx]) / 2)

            return np.array(intervalList)

        def __SeekMinError__(self,trainProcSet,funcMatrix):
            sList = [1, -1]
            global YLoc
            global weightLoc
            global featureNum

            minError = self._trainLen
            minS = 0
            minTheta = 0
            minSigmalUnOfError = 0
            minSigmaUn = 0
            minfeatureIdx = 0
            minRealError  = 0
            for featureIdx in range(featureNum):
                midInterval = self.__genTrainInterVal__(featureIdx)  # 生成待分割点中的中值，用于分割线定位 Theta self.midInterval
                for s in sList:
                    for theta in midInterval:
                        error = 0
                        sigmalUn = 0
                        sigmalUnOfError = 0
                        realError = 0
                        for loop in range(trainProcSet.shape[0]):
                            sigmalUn += trainProcSet[loop][weightLoc]
                            val = trainProcSet[loop][featureIdx] - theta
                            if (trainProcSet[loop][YLoc] != s * self.__sign__(val)):
                                error += trainProcSet[loop][weightLoc]
                                realError += 1
                                sigmalUnOfError += trainProcSet[loop][weightLoc]

                        if error < minError:
                            minError = error
                            minS = s
                            minTheta = theta
                            minSigmaUn = sigmalUn
                            minSigmalUnOfError = sigmalUnOfError
                            minRealError = realError
                            minfeatureIdx = featureIdx

            elpsilonT = minSigmalUnOfError / minSigmaUn
            if (minError != 0):
                DiamondT = math.sqrt((1 - elpsilonT) / elpsilonT)
            else:
                DiamondT = 999999

            for loop in range(trainProcSet.shape[0]):
                if (trainProcSet[loop][YLoc] != minS * self.__sign__(trainProcSet[loop][minfeatureIdx] - minTheta)):
                    print('weight1:', trainProcSet[loop][weightLoc])
                    trainProcSet[loop][weightLoc] *= DiamondT
                    print ('DiamondT:',DiamondT)
                    print ('weight2:',trainProcSet[loop][weightLoc],',*:',trainProcSet[loop][weightLoc] * DiamondT)
                else:
                    trainProcSet[loop][weightLoc] /= DiamondT


            alpha = math.log(DiamondT)
            #print('alpha:',alpha,'minRealError:',minRealError,'minS:',minS,'minTheta:',minTheta,'featureIdx:',minfeatureIdx)

            #print ('***** trainProcSecond ***********')
            print (trainProcSet)

            element = np.array([(alpha, minRealError, minS, minTheta, minfeatureIdx)])
            element = np.reshape(element, (1, 5))
            funcMatrix = np.concatenate((funcMatrix, element), axis=0)
            return funcMatrix


        def __sketchLine__(self,fig,ax,funcMatrix):
            '''
            for i in range(funcMatrix.shape[0]):
                thetaList = funcMatrix[i][3] * np.ones(2)
                Yaxix = np.array([-1, 1])
                if 0 == funcMatrix[i][4]:
                    ax.plot(thetaList, Yaxix)
                else:
                    ax.plot(Yaxix, thetaList)
            '''

            pointArray1_X1 = np.zeros((1))
            pointArray1_X2 = np.zeros((1))
            pointArray2_X1 = np.zeros((1))
            pointArray2_X2 = np.zeros((1))

            countX1 = 0
            countX2 = 0
            for loopX1 in range(100):
                for loopX2 in range(100):
                    X1 = loopX1 / 50 - 1.0
                    X2 = loopX2 / 50 - 1.0
                    G = 0
                    for funLoop in range(funcMatrix.shape[0]):
                        (alpha, s, theta) = funcMatrix[funLoop][0], funcMatrix[funLoop][2], funcMatrix[funLoop][3]
                        if (0 == funcMatrix[funLoop][4]):
                            g = alpha * self.__sign__(s * (X1 - theta))
                        else:
                            g = alpha * self.__sign__(s * (X2 - theta))
                        G += g

                    elementX1 = X1
                    elementX2 = X2
                    if (1 == self.__sign__(G)):
                        pointArray1_X1 = np.append(pointArray1_X1, elementX1)
                        pointArray1_X2 = np.append(pointArray1_X2, elementX2)
                        countX1 += 1
                        if (0 == countX1):
                            pointArray1_X1 = np.delete(pointArray1_X1,0,0)
                            pointArray1_X2 = np.delete(pointArray1_X2, 0, 0)
                    else:
                        pointArray2_X1 = np.append(pointArray2_X1, elementX1)
                        pointArray2_X2 = np.append(pointArray2_X2, elementX2)
                        countX2 += 1
                        if (0 == countX2):
                            pointArray2_X1 = np.delete(pointArray2_X1,0,0)
                            pointArray2_X2 = np.delete(pointArray2_X2, 0, 0)
            Y1 = 100 * np.ones(pointArray1_X1.size)
            Y2 = 100 * np.ones(pointArray2_X1.size)
            ax.scatter(pointArray1_X1, pointArray1_X2, Y1, c='gold', marker="o", alpha=0.05)
            ax.scatter(pointArray2_X1, pointArray2_X2, Y2, c='yellowgreen', marker="x", alpha=0.05)

        def __checkTrainResult__(self,funcMatrix,index,trainProcSet):
            '''
            error = 0
            for loopi in range(self.DataSet.shape[0]):
                G = 0
                for loopk in range(funcMatrix.shape[0]):
                    (alpha,s,theta) = funcMatrix[loopk][0],funcMatrix[loopk][2],funcMatrix[loopk][3]
                    if (0 == funcMatrix[loopk][4]):
                        g = alpha * s * (self.DataSet[loopi][0] - theta)
                    else:
                        g = alpha * s * (self.DataSet[loopi][1] - theta)
                    G += g
                if (self.DataSet[loopi][YLoc] != self.__sign__(G)):
                    print (self.DataSet[loopi][YLoc],G)
                    error += 1
            '''
            error = 0
            G = 0
            global featureNum

            print('**************    funcMatrix     ***************')
            print('alpha    errorRate    minS    minTheta    featureIdx')
            print ('funcMatrix:',funcMatrix)
            for dataLoop in range(self.DataSet.shape[0]):
                for funLoop in range(funcMatrix.shape[0]):
                    (alpha, s, theta) = funcMatrix[funLoop][0], funcMatrix[funLoop][2], funcMatrix[funLoop][3]
                    if (0 == funcMatrix[funLoop][4]):
                        g = alpha * self.__sign__(s * (self.DataSet[dataLoop][0] - theta))
                    else:
                        g = alpha * self.__sign__(s * (self.DataSet[dataLoop][1] - theta))
                    G += g
                G /= funcMatrix.shape[0]
                if (self.DataSet[dataLoop][YLoc] != self.__sign__(G)):
                    print (self.DataSet[dataLoop][YLoc],G)
                    error += 1
            print ('Train Count:',index,',accurate:',1 - error/self.DataSet.shape[0],'error:',error)
            pass

        def __addInitWeightColumn__(self):
            weight = np.ones(self.trainSet.shape[0]) / self.trainSet.shape[0]
            trainProcSet = np.column_stack((self.trainSet, weight))
            return trainProcSet

        def __train__(self):

            funcMatrix = np.zeros((1,5))
            loopCount = 0

            trainProcSet = self.__addInitWeightColumn__()

            for loop in range(self._trainNum):
                loopCount += 1
                #self.__genTrainSetWithBootstrapping__(loop)   # 利用Bootstrapping生成训练集 self.trainSet

                print (trainProcSet)
                funcMatrix = self.__SeekMinError__(trainProcSet,funcMatrix)  # 寻找 self.trainSet 中最小Error对应的s，theta

                if 1 == loopCount:
                    funcMatrix = np.delete(funcMatrix, 0, 0)

                fig, ax = plt.subplots(1, 1)
                fig, ax = self.__sketchFrame__(fig, ax, trainProcSet)

                if (loop == self._trainNum - 1):
                    self.__sketchLine__(fig,ax,funcMatrix)
                    plt.show()
                self.__checkTrainResult__(funcMatrix,loopCount,trainProcSet)
                #print('**************    funcMatrix     ***************')
                #print('alpha    errorRate    minS    minTheta    featureIdx')
                #print(funcMatrix)
                #plt.show()
                pass

            print ('**************    funcMatrix     ***************')
            print ('alpha    errorRate    minS    minTheta    featureIdx')
            print (funcMatrix)
            #self.__checkTrainResult__(funcMatrix,100)

            plt.show()


        def __sketchFrame__(self,fig,ax, trainProcSet):
            global weightLoc
            correctListX1 = []
            correctListX2 = []

            errorListX1 = []
            errorListX2 = []

            correctY = []
            errorY = []

            print (self.X1)
            for loop in range(self.Y.shape[0]):
                weight = 0
                for pointLoop in range(trainProcSet.shape[0]):
                    if (self.X1[loop] == trainProcSet[pointLoop][0]) and (self.X2[loop] == trainProcSet[pointLoop][1]):
                        weight = trainProcSet[pointLoop][weightLoc]

                size = 100 #10000 * pow(weight,2)
                print (loop,size)
                if self.Y[loop] == 1:
                    correctListX1.append(self.X1[loop])
                    correctListX2.append(self.X2[loop])
                    correctY.append(size)
                else:
                    errorListX1.append(self.X1[loop])
                    errorListX2.append(self.X2[loop])
                    errorY.append(size)

            correctArrayX1 = np.array(correctListX1)
            correctArrayX2 = np.array(correctListX2)
            errorArrayX1 = np.array(errorListX1)
            errorArrayX2 = np.array(errorListX2)
            correctY = np.array(correctY)
            errorY = np.array(errorY)

            ax.scatter(correctArrayX1, correctArrayX2, correctY, c='g', marker="o")
            ax.scatter(errorArrayX1, errorArrayX2, errorY, c='b', marker="x")

            plt.xlabel("X1")
            plt.ylabel("X2")
            fig.suptitle('Adaptive Boost ')

            return fig,ax

        def __sketchTrainDataMap__(self,X1,X2,Y,s,theta):
            correctListX1 = []
            correctListX2 = []

            errorListX1 = []
            errorListX2 = []

            correctY = []
            errorY   = []

            for loop in range(Y.shape[0]):
                if Y[loop] == 1:
                    correctListX1.append(X1[loop])
                    correctListX2.append(X2[loop])
                    correctY.append(100)
                else:
                    errorListX1.append(X1[loop])
                    errorListX2.append(X2[loop])
                    errorY.append(100)

            correctArrayX1 = np.array(correctListX1)
            correctArrayX2 = np.array(correctListX2)
            errorArrayX1   = np.array(errorListX1)
            errorArrayX2    = np.array(errorListX2)
            correctY       = np.array(correctY)
            errorY         = np.array(errorY)

            fig, ax = plt.subplots(1, 1)
            ax.scatter(correctArrayX1, correctArrayX2,correctY,c='g', marker="o")
            ax.scatter(errorArrayX1, errorArrayX2,errorY,c='b', marker="x")

            thetaList = theta * np.ones(2)
            Yaxix = np.array([-1,1])
            ax.plot(thetaList,Yaxix)
            print(thetaList)
            print(X1)


            plt.xlabel("X1")
            plt.ylabel("X2")
            fig.suptitle('Adaptive Boost ')

            plt.show()

    softMarginSVMObj = AdaptiveBoost()
    softMarginSVMObj.__generateTrainData__()
    softMarginSVMObj.__train__()


if __name__ == "__main__":
    DoAdaptiveBoost()
import numpy as np
import cvxopt as cvx
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

def DoSoftMarginSVM():

    class SoftMarginSVM():
        def __init__(self):
            #self._b = np.zeros(1,dtype=np.float64)
            self._w = np.zeros(2)
            self._trainLen = 1000

        def __sign__(self,a):
            if a >= 0.0:
                return 1
            elif a < 0.0:
                return -1

        def __generateTrainData__(self):
            X1 = np.random.uniform(-1.0,1.0,self._trainLen)
            X2 = np.random.uniform(-1.0,1.0,self._trainLen)
            B  = 0.2
            W1=  0.3
            W2=  0.8
            Y = np.zeros(self._trainLen)
            for loop in range(Y.shape[0]):
                #Y[loop] = self.__sign__(W1 * X1[loop]+ W2 * X2[loop] - B) # (x - a)^2 + (y -b)^2 - B 小于0，则在园内
                Y[loop] = self.__sign__(W1 * X1[loop]*X1[loop] + W2 * X2[loop] * X2[loop] - B) # (x - a)^2 + (y -b)^2 - B 小于0，则在园内


            # soft-margin SVM 之前的SVM都无法解决噪声
            noiseArray = np.random.rand(Y.shape[0])
            for loop in range(Y.shape[0]):
                if (noiseArray[loop]) > 0.9:
                    Y[loop] = -1 * Y[loop]


            Xarray = np.stack((X1, X2), axis=-1)

            self._X  = Xarray
            self._Y = Y
            self.__sketchTrainDataMap__()




        def __train__(self):
            C = 20
            for loopn in range(self._Y.shape[0]):
                for loopm in range(self._Y.shape[0]):
                    # (Z0,Z1,Z2) = Z = phy(x) = (1,x1,x2,x1*x1,x2*x2,x2*x2
                    Zn  = np.array([self._X[loopn][0],
                                    self._X[loopn][1],
                                    self._X[loopn][0]*self._X[loopn][0],
                                    self._X[loopn][1]*self._X[loopn][1],
                                    self._X[loopn][0] * self._X[loopn][1]])

                    Zm  = np.array([self._X[loopm][0],
                                    self._X[loopm][1],
                                    self._X[loopm][0]*self._X[loopm][0],
                                    self._X[loopm][1]*self._X[loopm][1],
                                    self._X[loopm][0] * self._X[loopm][1]])


                    row = self._Y[loopn] * self._Y[loopm] * np.dot(Zn,Zm)

                    if loopm == 0:
                        rowInMartrix = np.array([row])
                    else:
                        rowInMartrix = np.append(rowInMartrix, row)

                if loopn == 0:
                    A = np.array([rowInMartrix])
                else:
                    A = np.concatenate((A, [rowInMartrix]), axis=0)

            P = cvx.matrix(A)
            q_array = np.multiply(-1,np.ones((self._Y.shape[0],)))
            q = cvx.matrix(q_array)

            A_Matrix = cvx.matrix((np.array(self._Y)),(1,self._Y.shape[0]))
            b_Matrix = cvx.matrix(np.array([0.0]))

            #G = cvx.matrix(np.multiply(-1, np.identity(self._Y.shape[0])))
            #h = cvx.matrix(np.zeros((self._Y.shape[0],)))
            #sol = cvx.solvers.qp(P,q,G,h,A=A_Matrix, b=b_Matrix)  # 调用优化函数solvers.qp求解

            G = cvx.matrix(np.multiply(1, np.identity(self._Y.shape[0])))
            h = cvx.matrix(np.multiply(C,np.ones((self._Y.shape[0],))))

            sol = cvx.solvers.qp(P,q,G,h)  # 调用优化函数solvers.qp求解
            self.result = sol['x']
            print (type(self.result),self.result.size)
            self._a = np.array(self.result)
            print(type(self._a),self._a.shape)
            self._b = np.float64(self.result[0])

            w1 = 0.0
            w2 = 0.0
            w3 = 0.0
            w4 = 0.0
            w5 = 0.0
            for loop in range(self._Y.shape[0]):
                w1 += self._a[loop] * self._Y[loop] * self._X[loop][0]
                w2 += self._a[loop] * self._Y[loop] * self._X[loop][1]
                w3 += self._a[loop] * self._Y[loop] *  self._X[loop][0] * self._X[loop][0]
                w4 += self._a[loop] * self._Y[loop] * self._X[loop][1]*self._X[loop][1]
                w5 += self._a[loop] * self._Y[loop] * self._X[loop][0]*self._X[loop][1]

            print (w1.shape)
            self._w1 =  w1
            self._w2 =  w2
            self._w3 =  w3
            self._w4 =  w4
            self._w5 =  w5
            w = np.array([w1,w2,w3,w4,w5])
            w = np.reshape(w,(5))
            print(w.shape)
            for loop in range(self._Y.shape[0]):
                x = np.array([self._X[loop][0],
                              self._X[loop][1],
                              self._X[loop][0] * self._X[loop][0],
                              self._X[loop][1] * self._X[loop][1],
                              self._X[loop][0] * self._X[loop][1]])
                if self._a[loop] > 0:
                    print (x.shape)
                    # b = self._Y[loop] - np.dot(w,x)
                    b = C - self._a[loop]
                    break
            self._b = b
            self.__calcAccurateRatio__()
            #self.__sketchMap__()

        def __calcAccurateRatio__(self):
            x1Array,x2Array = np.split(self._X,self._X.shape[1],axis = 1)
            x1Array = np.reshape(x1Array,(self._X.shape[0]))
            x2Array = np.reshape(x2Array, (self._X.shape[0]))
            x3Array = np.multiply(x1Array,x1Array)
            x4Array = np.multiply(x2Array, x2Array)
            x5Array = np.multiply(x1Array, x2Array)
            optimalYArray = self._w1 * x1Array \
                            + self._w2 * x2Array \
                            + self._w3 * x3Array  \
                            + self._w4 * x4Array \
                            + self._w5 * x5Array \
                            + self._b
            sumCount   = 0
            accrateCnt = 0
            for loop in range(self._Y.shape[0]):
                sumCount += 1
                if self.__sign__(optimalYArray[loop]) == self._Y[loop]:
                    accrateCnt += 1
            print ('accurate ratio:',accrateCnt/sumCount)

        def __sketchMap__(self):
            correctListX1 = []
            correctListX2 = []

            errorListX1 = []
            errorListX2 = []

            correctY = []
            errorY   = []
            for loop in range(self._Y.shape[0]):
                if self._Y[loop] == 1:
                    correctListX1.append(self._X[loop][0])
                    correctListX2.append(self._X[loop][1])
                    correctY.append(1)
                else:
                    errorListX1.append(self._X[loop][0])
                    errorListX2.append(self._X[loop][1])
                    errorY.append(-1)

            correctArrayX1 = np.array(correctListX1)
            correctArrayX2 = np.array(correctListX2)
            errorArrayX1   = np.array(errorListX1)
            errorArrayX2    = np.array(errorListX2)
            correctY       = np.array(correctY)
            errorY         = np.array(errorY)

            optimalY = []

            for loop in range(self._X.shape[0]):
                val = self._w1 * self._X[loop][0] + self._w2 * self._X[loop][1]  +self._b
                optimalY.append(val)

            optimalYArray = np.array(optimalY)

            fig, ax = plt.subplots(1, 1)
            ax = fig.gca(projection='3d')
            ax.scatter(correctArrayX1, correctArrayX2,correctY,c='g', marker="x")
            ax.scatter(errorArrayX1, errorArrayX2,errorY,c='b', marker="+")

            x1Array,x2Array = np.split(self._X,self._X.shape[1],axis = 1)
            #ax.scatter(x1Array, x2Array, optimalYArray,c='orangered', marker=".")
            x1Array = np.reshape(x1Array,(self._X.shape[0]))
            x2Array = np.reshape(x2Array, (self._X.shape[0]))
            x3Array = np.multiply(x1Array,x1Array)
            x4Array = np.multiply(x2Array, x2Array)
            x5Array = np.multiply(x1Array, x2Array)
            print (type(x1Array),self._X.shape[0],type(x4Array),x4Array.shape)
            x1SplitResult = np.split(x1Array, [10, self._X.shape[0]],axis = 0)
            x2SplitResult = np.split(x2Array, [10, self._X.shape[0]],axis = 0)
            x3SplitResult = np.split(x3Array, [10, self._X.shape[0]],axis = 0)
            x4SplitResult = np.split(x4Array, [10, self._X.shape[0]],axis = 0)
            x5SplitResult = np.split(x5Array, [10, self._X.shape[0]], axis=0)
            # x1Array, x2Array = np.meshgrid(x1SplitResult[0], x2SplitResult[0])  # 将坐标向量变为坐标矩阵，列为x的长度，行为y的长度
            print(x1Array.shape,x2Array.shape,x3Array.shape)
            optimalYArray = self._w1 * x1Array \
                            + self._w2 * x2Array \
                            + self._w3 * x3Array  \
                            + self._w4 * x4Array \
                            + self._w5 * x5Array \
                            + self._b
            sumCount   = 0
            accrateCnt = 0
            for loop in range(self._Y.shape[0]):
                sumCount += 1
                if self.__sign__(optimalYArray[loop]) == self._Y[loop]:
                    accrateCnt += 1
            print ('accurate ratio:',accrateCnt/sumCount)
            ax.plot_surface(x1Array, x2Array, optimalYArray, rstride=1, cstride=1,color='coral', linewidth=0, antialiased=True,alpha=0.1)
            ax.set_xlabel("x1-label", color='r')
            ax.set_ylabel("x2-label", color='g')
            ax.set_zlabel("Y-label", color='b')

            ax.set_zlim3d(-1.5, 1.5)  # 设置z坐标轴

            fig.suptitle('Linear SVM ')
            plt.savefig("Linear_SVM.png")
            plt.show()

        def __sketchTrainDataMap__(self):
            correctListX1 = []
            correctListX2 = []

            errorListX1 = []
            errorListX2 = []

            correctY = []
            errorY   = []

            for loop in range(self._Y.shape[0]):
                if self._Y[loop] == 1:
                    correctListX1.append(self._X[loop][0])
                    correctListX2.append(self._X[loop][1])
                    correctY.append(1)
                else:
                    errorListX1.append(self._X[loop][0])
                    errorListX2.append(self._X[loop][1])
                    errorY.append(1)

            correctArrayX1 = np.array(correctListX1)
            correctArrayX2 = np.array(correctListX2)
            errorArrayX1   = np.array(errorListX1)
            errorArrayX2    = np.array(errorListX2)
            correctY       = np.array(correctY)
            errorY         = np.array(errorY)

            fig, ax = plt.subplots(1, 1)
            ax = fig.gca(projection='3d')
            ax.scatter(correctArrayX1, correctArrayX2,correctY,c='g', marker="x")
            ax.scatter(errorArrayX1, errorArrayX2,errorY,c='b', marker="+")
            fig.suptitle('Dual SVM ')

            plt.show()

    softMarginSVMObj = SoftMarginSVM()
    softMarginSVMObj.__generateTrainData__()
    softMarginSVMObj.__train__()

if __name__ == "__main__":
    DoSoftMarginSVM()
import numpy as np
import cvxopt as cvx
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d


def SolveQPWithCvxopt():
    P = cvx.matrix([[1.0, 0.0], [0.0, 0.0]])  # matrix里区分int和double，所以数字后面都需要加小数点
    q = cvx.matrix([3.0, 4.0])
    G = cvx.matrix([[-1.0, 0.0, -1.0, 2.0, 3.0], [0.0, -1.0, -3.0, 5.0, 4.0]])
    h = cvx.matrix([0.0, 0.0, -15.0, 100.0, 80.0])

    sol = cvx.solvers.qp(P, q, G, h)  # 调用优化函数solvers.qp求解
    print(sol['x'])  # sol['x'] 里面是对应的最优解

    P = cvx.matrix(np.diag([1.0, 0]))  # 对于一些特殊矩阵，用numpy创建会方便很多（在本例中可能感受不大）
    q = cvx.matrix(np.array([3.0, 4]))
    G = cvx.matrix(np.array([[-1.0, 0], [0, -1], [-1, -3], [2, 5], [3, 4]]))
    h = cvx.matrix(np.array([0.0, 0, -15, 100, 80]))
    sol = cvx.solvers.qp(P, q, G, h)
    print(sol['x'])  # 打印结果，sol里面还有很多其他属性，读者可以自行了解

def DoLinearSVM():

    class LinearSVM():
        def __init__(self):
            #self._b = np.zeros(1,dtype=np.float64)
            self._w = np.zeros(2)

        def __sign__(self,a):
            if a >= 0.0:
                return 1
            elif a < 0.0:
                return -1

        def __generateTrainData__(self):
            X1 = np.random.uniform(-1.0,1.0,200)
            X2 = np.random.uniform(-1.0,1.0,200)
            B  = 0.03
            W1=  0.3
            W2=  0.8
            Y = np.zeros(200)
            for loop in range(Y.shape[0]):
                Y[loop] = self.__sign__(W1 * X1[loop] + W2 * X2[loop] + B)

            '''
            noiseArray = np.random.rand(Y.shape[0])
            for loop in range(Y.shape[0]):
                if (noiseArray[loop]) > 0.9:
                    Y[loop] = -1 * Y[loop]
            '''

            Xarray = np.stack((X1, X2), axis=-1)

            self._X  = Xarray
            self._Y = Y
            self.sketchTrainDataMap()




        def train(self):
            Q = cvx.matrix(np.diag([0.0, 1.0,1.0])) # Q =[[0,0,0],[0,1,0],[0,0,1]]

            for loop in range(self._Y.shape[0]):
                if loop == 0:
                    A = np.array([[-1.0*self._Y[loop],
                                 np.multiply(self._Y[loop] *-1.0,self._X[loop][0]),
                                 np.multiply(self._Y[loop] *-1.0,self._X[loop][1])]])
                else:
                    rowArray = np.array([[-1.0*self._Y[loop],
                                 np.multiply(self._Y[loop] * -1.0,self._X[loop][0]),
                                 np.multiply(self._Y[loop] * -1.0,self._X[loop][1])]])
                    A = np.concatenate((A, rowArray), axis=0)

            cn_array = np.multiply(-1,np.ones((200,)))
            cn = cvx.matrix(cn_array)

            p = cvx.matrix([0.0, 0.0, 0.0])
            A = cvx.matrix(A)
            cn = cvx.matrix(cn)

            sol = cvx.solvers.qp(Q, p, A, cn)  # 调用优化函数solvers.qp求解
            self.result = sol['x']
            self._b = np.float64(self.result[0])
            self._w1 =  np.float64(self.result[1])
            self._w2 =  np.float64(self.result[2])
            self.sketchMap()


        def sketchMap(self):
            correctListX1 = []
            correctListX2 = []

            errorListX1 = []
            errorListX2 = []

            for loop in range(self._Y.shape[0]):
                if self._Y[loop] == 1:
                    correctListX1.append(self._X[loop][0])
                    correctListX2.append(self._X[loop][1])
                else:
                    errorListX1.append(self._X[loop][0])
                    errorListX2.append(self._X[loop][1])

            correctArrayX1 = np.array(correctListX1)
            correctArrayX2 = np.array(correctListX2)
            errorArrayX1   = np.array(errorListX1)
            errorArrayX2    = np.array(errorListX2)

            optimalY = []

            for loop in range(self._X.shape[0]):
                val = self._w1 * self._X[loop][0] + self._w2 * self._X[loop][1]+self._b
                optimalY.append(val)

            optimalYArray = np.array(optimalY)

            fig, ax = plt.subplots(1, 1)
            ax = fig.gca(projection='3d')
            ax.scatter(correctArrayX1, correctArrayX2,c='g', marker="x")
            ax.scatter(errorArrayX1, errorArrayX2,c='b', marker="+")

            x1Array,x2Array = np.split(self._X,self._X.shape[1],axis = 1)
            #ax.scatter(x1Array, x2Array, optimalYArray,c='orangered', marker=".")
            x1Array = np.reshape(x1Array,(self._X.shape[0]))
            x2Array = np.reshape(x2Array, (self._X.shape[0]))
            print (type(x1Array),self._X.shape[0],x1Array.shape)
            x1SplitResult = np.split(x1Array, [10, self._X.shape[0]],axis = 0)
            x2SplitResult = np.split(x2Array, [10, self._X.shape[0]],axis = 0)
            x1Array, x2Array = np.meshgrid(x1SplitResult[0], x2SplitResult[0])  # 将坐标向量变为坐标矩阵，列为x的长度，行为y的长度
            print(x1Array.shape,x2Array.shape)
            optimalYArray = self._w1 * x1Array + self._w2 * x2Array +self._b
            ax.plot_surface(x1Array, x2Array, optimalYArray, rstride=1, cstride=1,color='coral', linewidth=0, antialiased=True,alpha=0.1)
            ax.set_xlabel("x1-label", color='r')
            ax.set_ylabel("x2-label", color='g')
            ax.set_zlabel("Y-label", color='b')

            ax.set_zlim3d(-50, 50)  # 设置z坐标轴

            fig.suptitle('Linear SVM ')
            plt.savefig("Linear_SVM.png")
            plt.show()

        def sketchTrainDataMap(self):
            correctListX1 = []
            correctListX2 = []

            errorListX1 = []
            errorListX2 = []

            for loop in range(self._Y.shape[0]):
                if self._Y[loop] == 1:
                    correctListX1.append(self._X[loop][0])
                    correctListX2.append(self._X[loop][1])
                else:
                    errorListX1.append(self._X[loop][0])
                    errorListX2.append(self._X[loop][1])

            correctArrayX1 = np.array(correctListX1)
            correctArrayX2 = np.array(correctListX2)
            errorArrayX1   = np.array(errorListX1)
            errorArrayX2    = np.array(errorListX2)


            fig, ax = plt.subplots(1, 1)
            ax = fig.gca(projection='3d')
            ax.scatter(correctArrayX1, correctArrayX2,c='g', marker="x")
            ax.scatter(errorArrayX1, errorArrayX2,c='b', marker="+")
            fig.suptitle('Linear SVM ')

            plt.show()

        def loadTrainData(self,fileLocation):
            data = []
            label = []
            lineNum = 0
            with open(fileLocation) as f:
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
            self._X  = dataArray
            self._Y = labelArray

    linearSVMObj = LinearSVM()
    linearSVMObj.__generateTrainData__()
    #linearSVMObj.loadTrainData( 'G:\\林轩田教程\\MachineLearningFoundations\\homework5\\data\\question13_TRAIN.txt')
    linearSVMObj.train()


def DoDualSVM():

    class DualSVM():
        def __init__(self):
            #self._b = np.zeros(1,dtype=np.float64)
            self._w = np.zeros(2)
            self._trainLen = 10

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
                Y[loop] = self.__sign__(W1 * X1[loop]*X1[loop] + W2 * X2[loop] * X2[loop] - B) # (x - a)^2 + (y -b)^2 - B 小于0，则在园内

            '''
            noiseArray = np.random.rand(Y.shape[0])
            for loop in range(Y.shape[0]):
                if (noiseArray[loop]) > 0.9:
                    Y[loop] = -1 * Y[loop]
            '''

            Xarray = np.stack((X1, X2), axis=-1)

            self._X  = Xarray
            self._Y = Y
            self.__sketchTrainDataMap__()




        def __train__(self):
            for loopn in range(self._Y.shape[0]):
                for loopm in range(self._Y.shape[0]):
                    Zn  = np.array([self._X[loopn][0],
                                    self._X[loopn][1]])
                    Zm  = np.array([self._X[loopm][0],
                                    self._X[loopm][1]])


                    row = self._Y[loopn] * self._Y[loopm] * np.dot(Zn,Zm)

                    if loopm == 0:
                        rowInMartrix = np.array([row])
                    else:
                        rowInMartrix = np.append(rowInMartrix, row)

                if loopn == 0:
                    A = np.array([rowInMartrix])
                else:
                    A = np.concatenate((A, [rowInMartrix]), axis=0)
                print (A.shape)
            '''
            for loopn in range(self._Y.shape[0]):
                row = np.ones([self._Y.shape[0],2])
                print (row.shape)
                Zn = np.array([self._X[loopn][0],
                               self._X[loopn][1]])
                row = np.multiply(row,Zn)
                row = np.multiply(row,self._Y[loopn])
                print(row.shape)
                if loopn == 0:
                    ZnArray = np.array([row])
                else:
                    ZnArray = np.concatenate((ZnArray, [row]), axis=0)
            print('ZnArrayShape:', ZnArray.shape)

            for loopm in range(self._Y.shape[0]):
                row = np.array([[self._X[loopm][0],self._X[loopm][1]]])
                row = np.multiply(row, self._Y[loopm])
                if loopm == 0:
                    ZmArray = np.array([row])
                else:
                    ZmArray = np.concatenate((ZmArray, [row]), axis=0)
            print ('ZmArrayShape:',ZmArray.shape)

            array = np.dot(ZnArray,Zm)
            print (array.shape)
            '''
            P = cvx.matrix(A)
            q_array = np.multiply(-1,np.ones((self._Y.shape[0],)))
            q = cvx.matrix(q_array)

            for loop in range(self._Y.shape[0]):
                if 0 == loop:
                    G_array = np.array([self._Y])
                else:
                    G_array = np.concatenate((G_array, [self._Y]), axis=0)

            print (np.transpose(np.array(self._Y)).shape)
            A_Matrix = cvx.matrix((np.array(self._Y)),(1,10))
            b_Matrix = cvx.matrix(np.array([0.0]))

            G = cvx.matrix(np.multiply(-1, np.ones((self._Y.shape[0],))))
            h = cvx.matrix(np.zeros((self._Y.shape[0],)))

            for loopn in range(self._Y.shape[0]):
                row  = np.multiply(-1, np.ones((self._Y.shape[0],)))

                if 0 == loopn:
                    G_array = np.array([row])
                else:
                    G_array = np.concatenate((G_array, [row]), axis=0)

            print (G_array.shape)
            G_array = cvx.matrix(G_array)
            print ('**** q ****')
            print(q)
            print('**** G ****')
            print(G)
            print('**** h ****')
            print(h)
            print(type(P),type(q),type(G),type(h),type(A_Matrix),type(A_Matrix))
            sol = cvx.solvers.qp(P,q,G_array,h,A=A_Matrix, b=b_Matrix)  # 调用优化函数solvers.qp求解
            self.result = sol['x']
            self._b = np.float64(self.result[0])
            self._w1 =  np.float64(self.result[1])
            self._w2 =  np.float64(self.result[2])
            self.__sketchMap__()


        def __sketchMap__(self):
            correctListX1 = []
            correctListX2 = []

            errorListX1 = []
            errorListX2 = []

            for loop in range(self._Y.shape[0]):
                if self._Y[loop] == 1:
                    correctListX1.append(self._X[loop][0]*self._X[loop][0])
                    correctListX2.append(self._X[loop][1]*self._X[loop][1])
                else:
                    errorListX1.append(self._X[loop][0]*self._X[loop][0])
                    errorListX2.append(self._X[loop][1]*self._X[loop][1])

            correctArrayX1 = np.array(correctListX1)
            correctArrayX2 = np.array(correctListX2)
            errorArrayX1   = np.array(errorListX1)
            errorArrayX2    = np.array(errorListX2)

            optimalY = []

            for loop in range(self._X.shape[0]):
                val = self._w1 * self._X[loop][0]*self._X[loop][0] + self._w2 * self._X[loop][1] * self._X[loop][1]  +self._b
                optimalY.append(val)

            optimalYArray = np.array(optimalY)

            fig, ax = plt.subplots(1, 1)
            ax = fig.gca(projection='3d')
            ax.scatter(correctArrayX1, correctArrayX2,c='g', marker="x")
            ax.scatter(errorArrayX1, errorArrayX2,c='b', marker="+")

            x1Array,x2Array = np.split(self._X,self._X.shape[1],axis = 1)
            #ax.scatter(x1Array, x2Array, optimalYArray,c='orangered', marker=".")
            x1Array = np.reshape(x1Array,(self._X.shape[0]))
            x2Array = np.reshape(x2Array, (self._X.shape[0]))
            print (type(x1Array),self._X.shape[0],x1Array.shape)
            x1SplitResult = np.split(x1Array, [10, self._X.shape[0]],axis = 0)
            x2SplitResult = np.split(x2Array, [10, self._X.shape[0]],axis = 0)
            x1Array, x2Array = np.meshgrid(x1SplitResult[0], x2SplitResult[0])  # 将坐标向量变为坐标矩阵，列为x的长度，行为y的长度
            print(x1Array.shape,x2Array.shape)
            optimalYArray = self._w1 * x1Array + self._w2 * x2Array +self._b
            ax.plot_surface(x1Array, x2Array, optimalYArray, rstride=1, cstride=1,color='coral', linewidth=0, antialiased=True,alpha=0.1)
            ax.set_xlabel("x1-label", color='r')
            ax.set_ylabel("x2-label", color='g')
            ax.set_zlabel("Y-label", color='b')

            ax.set_zlim3d(-50, 50)  # 设置z坐标轴

            fig.suptitle('Linear SVM ')
            plt.savefig("Linear_SVM.png")
            plt.show()

        def __sketchTrainDataMap__(self):
            correctListX1 = []
            correctListX2 = []

            errorListX1 = []
            errorListX2 = []

            for loop in range(self._Y.shape[0]):
                if self._Y[loop] == 1:
                    correctListX1.append(self._X[loop][0])
                    correctListX2.append(self._X[loop][1])
                else:
                    errorListX1.append(self._X[loop][0])
                    errorListX2.append(self._X[loop][1])

            correctArrayX1 = np.array(correctListX1)
            correctArrayX2 = np.array(correctListX2)
            errorArrayX1   = np.array(errorListX1)
            errorArrayX2    = np.array(errorListX2)


            fig, ax = plt.subplots(1, 1)
            ax = fig.gca(projection='3d')
            ax.scatter(correctArrayX1, correctArrayX2,c='g', marker="x")
            ax.scatter(errorArrayX1, errorArrayX2,c='b', marker="+")
            fig.suptitle('Linear SVM ')

            plt.show()

    dualSVMObj = DualSVM()
    dualSVMObj.__generateTrainData__()
    dualSVMObj.__train__()

def SolveQPExample():
    X = np.array([[0,0],[2,2],[2,0],[3,0]])
    y = np.array([-1,-1,+1,+1])
    Qd = np.array([[0.0,0.0,0.0,0.0],[0.0,8.0,-4.0,-6.0],[0.0,-4.0,4.0,6.0],[0.0,-6.0,6.0,9.0]])
    print (Qd.shape)
    Qd = cvx.matrix(Qd)
    Ad = np.array([[-1.0,-1.0,1.0,1.0],[1.0,1.0,-1.0,-1.0],[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]])
    Ad = np.multiply(-1,Ad)
    Ad = cvx.matrix(Ad)

    q_array = np.multiply(-1, np.ones((4,)))
    q = cvx.matrix(q_array)

    h_array = np.zeros((4,))
    h = cvx.matrix(h_array)
    sol = cvx.solvers.qp(Qd, q, Ad, h)  # 调用优化函数solvers.qp求解
    print (sol)


if __name__ == "__main__":
    #SolveQPExample()
    SolveQPWithCvxopt()
    DoDualSVM()

from PSO import PSO
import numpy as np
from tools import func_transformer
from base import SkoBase
import os
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

def MCC(y_ture, y_pred):
    mcc = metrics.matthews_corrcoef(y_ture, y_pred)
    return mcc
Algorith=['SAMME.R','SAMME']

def demo_func1(x):
    flag = 0
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    X=x
    for i in range(len(X)):
        if X[i]<0:
            X[i]=-X[i]
    X[1] = int(X[1]*100)
    X[2] = int(X[2])
    X[3] = int(X[3]*10)
    while X[0] > 1:
        X[0] = X[0] - 1
    while X[1] >100:
        X[1] = X[1] - 99
    while X[2] > 1:
        X[2] = X[0] - 1
    while X[3] > 10:
        X[3] = X[3] - 10

    fileList = os.listdir("D:\PycharmWorkspace\\thirdteenth\sel_Data\\train\positive")
    flag = 0
    for name in fileList:
        flag += 1

        x_train.append(np.load("D:\PycharmWorkspace\\thirdteenth\sel_Data\\train\positive\\" + name))
        # print(np.load("D:\PycharmWorkspace\zyh\Data\\train\positive\\"+name))
        y_train.append(1)
    fileList = os.listdir("D:\PycharmWorkspace\\thirdteenth\sel_Data\\train\\negitive")
    flag = 0
    for name in fileList:
        flag += 1
        x_train.append(np.load("D:\PycharmWorkspace\\thirdteenth\sel_Data\\train\\negitive\\" + name))
        y_train.append(0)
    fileList = os.listdir("D:\PycharmWorkspace\\thirdteenth\sel_Data\\test\positive")
    flag = 0
    for name in fileList:
        flag += 1
        x_test.append(np.load("D:\PycharmWorkspace\\thirdteenth\sel_Data\\test\positive\\" + name))
        y_test.append(1)
    fileList = os.listdir("D:\PycharmWorkspace\\thirdteenth\sel_Data\\test\\negitive")
    flag = 0
    for name in fileList:
        flag += 1
        x_test.append(np.load("D:\PycharmWorkspace\\thirdteenth\sel_Data\\test\\negitive\\" + name))
        y_test.append(0)
    kf = KFold(n_splits=10)
    kf.get_n_splits(x_train)
    mcc1 = 0
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    for train_index, test_index in kf.split(x_train):
        x1_train = []
        y1_train = []
        x1_test = []
        y1_test = []
        # clf = svm.SVC()
        # print("TRAIN:", train_index, "TEST:", test_index)
        for i in range(len(train_index)):
            x1_train.append(x_train[train_index[i]])
            y1_train.append(y_train[train_index[i]])
        # print(len(y1_train))
        # print(y1_train)
        for i in range(len(test_index)):
            x1_test.append(x_train[test_index[i]])
            y1_test.append(y_train[test_index[i]])
        # print(y1_test)
        x1_train = np.array(x1_train)
        y1_train = np.array(y1_train)
        x1_test = np.array(x1_test)
        y1_test = np.array(y1_test)

        clf = AdaBoostClassifier(learning_rate=X[0], n_estimators=int(X[1]),algorithm=Algorith[int(X[2])],random_state=int(X[3]))
        # clf = XGBClassifier(l)
        clf.fit(x1_train, y1_train)
        y_predict = clf.predict(x1_test)
        #       y_pred_probability = clf.predict_proba(x1_test)

        # print(y_pred_probability)

        # df2 = pd.DataFrame(y_pred_probability)
        mcc = MCC(y1_test, y_predict)
        mcc1 += mcc

    print(X)
    mcc1 = mcc1 / 10

    print(mcc1)

    return mcc1
'''
def demo_func(x):
    x1, x2 = x
    return x1 ** 2 + (x2 - 0.05) ** 2
'''
pso = PSO(func=demo_func1, dim=4, pop=100, max_iter=200)
pso.record_mode = True
pso.run()
print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

# %% Now Plot the animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

record_value = pso.record_value
X_list, V_list = record_value['X'], record_value['V']

fig, ax = plt.subplots(1, 1)
ax.set_title('title', loc='center')
line = ax.plot([], [], 'b.')

X_grid, Y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 40), np.linspace(-1.0, 1.0, 40))
Z_grid = demo_func((X_grid, Y_grid))
ax.contour(X_grid, Y_grid, Z_grid, 20)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

plt.ion()
p = plt.show()


def update_scatter(frame):
    i, j = frame // 10, frame % 10
    ax.set_title('iter = ' + str(i))
    X_tmp = X_list[i] + V_list[i] * j / 10.0
    plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
    return line


ani = FuncAnimation(fig, update_scatter, blit=True, interval=25, frames=300)
plt.show()

# ani.save('pso.gif', writer='pillow')
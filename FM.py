from __future__ import division
from math import exp
from numpy import *
import numpy as np
from random import normalvariate
import pandas as pd
# reference
# https://segmentfault.com/a/1190000020254554
# https://blog.csdn.net/john_xyz/article/details/78933253

train_data = 'fm_data/train.txt'
test_data = 'fm_data/test.txt'

def preprocess(data):
    feature=np.array(data.iloc[:,:-1])
    label=data.iloc[:,-1].map(lambda x: 1 if x==1 else -1)
    zmax, zmin = feature.max(axis=0), feature.min(axis=0)
    feature = (feature - zmin) / (zmax - zmin)
    label=np.array(label)
    return feature,label

def sigmoid(x):
    return 1.0 / (1 + exp(-x))

def SGD_FM(data, label, k, iter):
    m, num_feature = shape(data)
    alpha = 0.01
    w = zeros((num_feature, 1))      #一阶特征的系数
    w_0 = 0.0
    v = normalvariate(0, 0.2) * ones((num_feature, k))   #即生成辅助向量，用来训练二阶交叉特征的系数

    for it in range(iter):
        true_loss = 0.0
        false_loss = 0.0
        for x in range(m):
            fm_1 = data[x] * v
            fm_2 = multiply(data[x], data[x]) * multiply(v, v)
            fm = sum(multiply(fm_1, fm_1) - fm_2) / 2.
            y = w_0 + data[x] * w + fm  # FM预测输出
            loss = sigmoid(label[x] * y[0, 0])-1    #真实损失过大，与真实损失相像。loss针对当前样本，不做为最后loss衡量
            true_loss += -np.log(sigmoid(label[x] * y[0, 0]))
            false_loss += -loss
            w_0 -= alpha * loss * label[x]
            for i in range(num_feature):
                if data[x, i] != 0:
                    w[i, 0] -= alpha * loss * label[x] * data[x, i]
                    for j in range(k):
                        v[i, j] -= alpha * loss * label[x] * (data[x, i] * fm_1[0, j] - v[i, j] * data[x, i] * data[x, i])
        print("第{}次迭代后真损失为{}，假损失为{}".format(it, true_loss, false_loss))
    return w_0, w, v

def predict(data, label, w_0, w, v):
    m, _ = shape(data)
    err = 0
    num = 0
    result = []
    for x in range(m):   #计算每一个样本的误差
        num += 1
        fm_1 = data[x] * v
        fm_2 = multiply(data[x], data[x]) * multiply(v, v)
        fm = sum(multiply(fm_1, fm_1) - fm_2) / 2.
        y = w_0 + data[x] * w + fm  # 计算预测的输出
        prop = sigmoid(y[0, 0])
        result.append(prop)
        if prop < 0.5 and label[x] == 1.0:
            err += 1
        elif prop >= 0.5 and label[x] == -1.0:
            err += 1
        else:
            continue
    return float(err) / num

if __name__ == '__main__':
    train = pd.read_csv(train_data)
    test = pd.read_csv(test_data)
    x_train, y_train = preprocess(train)
    x_test, y_test = preprocess(test)
    w_0, w, v = SGD_FM(mat(x_train), y_train, 20, 200)
    print("训练集acc：%f" % (1 - predict(mat(x_train), y_train, w_0, w, v)))
    print("测试集acc：%f" % (1 - predict(mat(x_test), y_test, w_0, w, v)))

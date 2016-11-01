# encoding=utf-8
# 训练的整个过程

import jieba
from numpy import *
import numpy as np

fkeywords = open('../data/keywords_filtered.txt', 'r')

# matrix = zeros((1000, 20000))  # 初始化1000行,2W列的大矩阵

# from sklearn.svm import SVC
#
# # 初始化SVM
# clf1 = SVC()
# clf2 = SVC()
# clf3 = SVC()

# from sklearn.linear_model import LogisticRegression
#
# clf1 = LogisticRegression()
# clf2 = LogisticRegression()
# clf3 = LogisticRegression()

from sklearn.linear_model import SGDClassifier

clf1 = SGDClassifier()
clf2 = SGDClassifier()
clf3 = SGDClassifier()

# 降维
from sklearn.decomposition import PCA

COMPONENT_NUM = 1000  # 设置pca降维的维度值
pca = PCA(n_components=COMPONENT_NUM, whiten=True)

keywordList = []

for keyword in fkeywords.readlines():
    keywordList.append(str(keyword.strip('\n')))

print(keywordList)  # 大关键词列表

from itertools import islice

for batch in range(17):
    ftrainingData = open('../data/filter_zero.csv', 'r')
    print "sunpeng{0}".format(batch)
    matrix = zeros((1000, 20000))
    # batch 为 训练集分成20批次的每一批次的批次号
    x_train = []
    y_train = []
    begin_x = batch * 1000  # 初始化矩阵索引
    # 遍历训练集的每一行数据
    for line in islice(ftrainingData, begin_x, None):
        if begin_x >= (batch + 1) * 1000:
            break
        print "训练数据的行数{0}".format(begin_x)
        begin_y = 0
        list = line.split('\t')
        trainList = list[4:]
        temp_y = list[1:4]  # 对每一行的label值进行逐行添加
        y_train.append(temp_y)
        tempSet = set()  # 封装了每一行的所有关键字的集合
        for trainword in trainList:
            # 将每一行数据进行结巴分词
            seg_list = jieba.cut_for_search(str(trainword))
            tempSet = tempSet | set(seg_list)
        for keyword in keywordList:
            # 遍历关键词列表
            if keyword in tempSet:
                matrix[begin_x - 1000 * batch][begin_y] = 1
            else:
                begin_y = begin_y + 1
        begin_x = begin_x + 1
    print matrix.shape
    print np.array(y_train)
    print "PCA......{0}".format(batch)
    # pca.fit(matrix)
    # train_data = pca.transform(matrix)  # Fit the model with X and 在X上完成降维.
    # np.savetxt("train_data_{0}".format(batch), np.array(y_train))
    print "training......{0}".format(batch)
    clf1.fit(np.array(matrix), np.array(y_train)[:, 0])  # 训练
    clf2.fit(np.array(matrix), np.array(y_train)[:, 1])  # 训练
    clf3.fit(np.array(matrix), np.array(y_train)[:, 2])  # 训练
    print "training y...{0}--{1}--{2}".format(np.array(y_train)[:, 0], np.array(y_train)[:, 1], np.array(y_train)[:, 2])

print "train all over-----"

# 保存模型到本地
from sklearn.externals import joblib

joblib.dump(clf1, "train_model1_lr.m_111")
joblib.dump(clf2, "train_model2_lr.m_111")
joblib.dump(clf3, "train_model3_lr.m_111")

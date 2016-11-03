# encoding=utf-8
# 训练的整个过程

import jieba
from numpy import *
import numpy as np

fkeywords = open('../data/keywords_filtered_feature_5.txt', 'r')

from sklearn.linear_model import SGDClassifier

clf1 = SGDClassifier()
clf2 = SGDClassifier()
clf3 = SGDClassifier()

from sklearn.svm import SVC

clf4 = SVC()
clf5 = SVC()
clf6 = SVC()

from sklearn.ensemble import RandomForestClassifier

clf7 = RandomForestClassifier()
clf8 = RandomForestClassifier()
clf9 = RandomForestClassifier()

keywordList = []

for keyword in fkeywords.readlines():
    keywordList.append(str(keyword.strip('\n')))

print(keywordList)  # 大关键词列表

from itertools import islice

for batch in range(1):
    ftrainingData = open('../data/sunpengtest.csv', 'r')
    print "sunpeng{0}".format(batch)
    matrix = zeros((10000, 1000))
    # batch 为 训练集分成17批次的每一批次的批次号
    x_train = []
    y_train = []
    begin_x = batch * 10000  # 初始化矩阵索引
    # 遍历训练集的每一行数据
    for line in islice(ftrainingData, begin_x, None):
        if begin_x >= (batch + 1) * 10000:
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
                matrix[begin_x - 10000 * batch][begin_y] = 1
            else:
                begin_y = begin_y + 1
        begin_x = begin_x + 1
    print matrix.shape
    print np.array(y_train)

    print "training......{0}".format(batch)
    clf1.fit(np.array(matrix), np.array(y_train)[:, 0])  # 训练
    clf2.fit(np.array(matrix), np.array(y_train)[:, 1])  # 训练
    clf3.fit(np.array(matrix), np.array(y_train)[:, 2])  # 训练
    clf4.fit(np.array(matrix), np.array(y_train)[:, 0])
    clf5.fit(np.array(matrix), np.array(y_train)[:, 1])
    clf6.fit(np.array(matrix), np.array(y_train)[:, 2])
    clf7.fit(np.array(matrix), np.array(y_train)[:, 0])
    clf8.fit(np.array(matrix), np.array(y_train)[:, 1])
    clf9.fit(np.array(matrix), np.array(y_train)[:, 2])
    print "training y...{0}--{1}--{2}".format(np.array(y_train)[:, 0], np.array(y_train)[:, 1], np.array(y_train)[:, 2])
    print clf1.score(np.array(matrix), np.array(y_train)[:, 0])
    print clf2.score(np.array(matrix), np.array(y_train)[:, 1])
    print clf3.score(np.array(matrix), np.array(y_train)[:, 2])
    print clf4.score(np.array(matrix), np.array(y_train)[:, 0])
    print clf5.score(np.array(matrix), np.array(y_train)[:, 1])
    print clf6.score(np.array(matrix), np.array(y_train)[:, 2])
    print clf7.score(np.array(matrix), np.array(y_train)[:, 0])
    print clf8.score(np.array(matrix), np.array(y_train)[:, 1])
    print clf9.score(np.array(matrix), np.array(y_train)[:, 2])
    print 'score~~~~~~~'

print "train all over-----"

# # 保存模型到本地
# from sklearn.externals import joblib
#
# joblib.dump(clf1, "train_model1_lr.m_111")
# joblib.dump(clf2, "train_model2_lr.m_111")
# joblib.dump(clf3, "train_model3_lr.m_111")
allLabels = 5000
right_1 = 0
right_2 = 0
right_3 = 0
labelsAll = []
from sklearn.externals import joblib

# # 加载本地模型
# clf1 = joblib.load("../model/train_model1_lr.m_112")
# clf2 = joblib.load("../model/train_model2_lr.m_112")
# clf3 = joblib.load("../model/train_model3_lr.m_112")

for batch in range(1):
    matrix = zeros((5000, 1000), dtype=int8)  # 初始化2W行,2W列的大矩阵
    x_train = []
    y_train = []
    ids = []
    begin_x = batch * 5000  # 初始化矩阵索引
    # input_file = open('../data/user_tag_query.2W.TEST')
    input_file = open('../data/sunpeng111.csv')
    for line in islice(input_file, begin_x, None):
        if begin_x >= (batch + 1) * 5000:
            break
        print "测试数据的行数{0}".format(begin_x)
        begin_y = 0
        list = line.split('\t')
        testList = list[4:]
        id = list[0:1]
        labels = list[1:4]
        ids.append(id)
        labelsAll.append(labels)
        tempSet = set()  # 封装了每一行的所有关键字的集合
        for testword in testList:
            # 将每一行数据进行结巴分词
            seg_list = jieba.cut_for_search(str(testword))
            tempSet = tempSet | set(seg_list)
        for keyword in keywordList:
            # 遍历关键词列表
            if keyword in tempSet:
                matrix[begin_x - 5000 * batch][begin_y] = 1
            else:
                begin_y = begin_y + 1
        begin_x = begin_x + 1

    # 降维
    print ids
    print "predict......{0}".format(batch)
    # pca.fit(matrix)
    # train_data = pca.transform(matrix)  # Fit the model with X and 在X上完成降维.
    print "pca over~{0}".format(batch)
    count = 0

    for x_test in np.array(matrix):
        thisDoc1 = array(x_test).reshape((1, -1))
        y_predTemp1 = clf1.predict(thisDoc1)
        y_predTemp2 = clf2.predict(thisDoc1)
        y_predTemp3 = clf3.predict(thisDoc1)
        y_predTemp4 = clf4.predict(thisDoc1)
        y_predTemp5 = clf5.predict(thisDoc1)
        y_predTemp6 = clf6.predict(thisDoc1)
        y_predTemp7 = clf7.predict(thisDoc1)
        y_predTemp8 = clf8.predict(thisDoc1)
        y_predTemp9 = clf9.predict(thisDoc1)
        print y_predTemp1
        print y_predTemp2
        print y_predTemp3
        print y_predTemp4
        print y_predTemp5
        print y_predTemp6
        print y_predTemp7
        print y_predTemp8
        print y_predTemp9

        # 投票选举机制
        from collections import Counter

        blist = [str(y_predTemp2[0]), str(y_predTemp5[0]), str(y_predTemp8[0])]
        word_countsb = Counter(blist)
        top = word_countsb.most_common(1)
        b = top[0][0]

        alist = [str(y_predTemp1[0]), str(y_predTemp4[0]), str(y_predTemp7[0])]
        word_countsb = Counter(alist)
        top = word_countsb.most_common(1)
        a = top[0][0]

        clist = [str(y_predTemp3[0]), str(y_predTemp6[0]), str(y_predTemp9[0])]
        word_countsb = Counter(clist)
        top = word_countsb.most_common(1)
        c = top[0][0]

        print labelsAll[count]
        print '---------'
        if a == labelsAll[count][0]:
            right_1 += 1
        if b == labelsAll[count][1]:
            right_2 += 1
        if c == labelsAll[count][2]:
            right_3 += 1
        count += 1

print right_1
print right_2
print right_3
print allLabels
print right_1 / float(allLabels)
print right_2 / float(allLabels)
print right_3 / float(allLabels)
print (right_3 + right_2 + right_1) / float(15000)

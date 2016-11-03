# encoding=utf-8
# 训练的整个过程

import jieba
from numpy import *
import numpy as np

fkeywords = open('../data/keywords_filtered_feature_4.txt', 'r')

from sklearn.svm import SVC

# 初始化SVM
clf1 = SVC()
clf2 = SVC()
clf3 = SVC()

# from sklearn.linear_model import LogisticRegression
#
# clf1 = LogisticRegression()
# clf2 = LogisticRegression()
# clf3 = LogisticRegression()

# from sklearn.linear_model import SGDClassifier
#
# clf1 = SGDClassifier()
# clf2 = SGDClassifier()
# clf3 = SGDClassifier()
#
# from sklearn.cluster import KMeans
#
# clf1 = KMeans(n_clusters=6, random_state=0)
# clf2 = KMeans(n_clusters=2, random_state=0)
# clf3 = KMeans(n_clusters=6, random_state=0)

# # 降维
# from sklearn.decomposition import PCA
#
# COMPONENT_NUM = 1000  # 设置pca降维的维度值
# pca = PCA(n_components=COMPONENT_NUM, whiten=True)

keywordList = []

for keyword in fkeywords.readlines():
    keywordList.append(str(keyword.strip('\n')))

print(keywordList)  # 大关键词列表

from itertools import islice

for batch in range(3):
    ftrainingData = open('../data/filter_zero.csv', 'r')
    print "sunpeng{0}".format(batch)
    matrix = zeros((5000, 2500))
    # batch 为 训练集分成17批次的每一批次的批次号
    x_train = []
    y_train = []
    begin_x = batch * 5000  # 初始化矩阵索引
    # 遍历训练集的每一行数据
    for line in islice(ftrainingData, begin_x, None):
        if begin_x >= (batch + 1) * 5000:
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
                matrix[begin_x - 5000 * batch][begin_y] = 1
            else:
                begin_y = begin_y + 1
        begin_x = begin_x + 1
    print matrix.shape
    print np.array(y_train)
    # print "PCA......{0}".format(batch)
    # pca.fit(matrix)
    # train_data = pca.transform(matrix)  # Fit the model with X and 在X上完成降维.
    # np.savetxt("train_data_{0}".format(batch), np.array(y_train))
    print "training......{0}".format(batch)
    clf1.fit(np.array(matrix), np.array(y_train)[:, 0])  # 训练
    clf2.fit(np.array(matrix), np.array(y_train)[:, 1])  # 训练
    clf3.fit(np.array(matrix), np.array(y_train)[:, 2])  # 训练
    print "training y...{0}--{1}--{2}".format(np.array(y_train)[:, 0], np.array(y_train)[:, 1], np.array(y_train)[:, 2])
    print clf1.score(np.array(matrix), np.array(y_train)[:, 0])
    print clf2.score(np.array(matrix), np.array(y_train)[:, 1])
    print clf3.score(np.array(matrix), np.array(y_train)[:, 2])
    print 'score~~~~~~~'

print "train all over-----"

# # 保存模型到本地
# from sklearn.externals import joblib
#
# joblib.dump(clf1, "train_model_112_a.m")
# joblib.dump(clf2, "train_model_112_b.m")
# joblib.dump(clf3, "train_model_112_b.m")



from itertools import islice
import codecs


with codecs.open('predict_113.csv', 'w', 'gbk') as writer:
    # 加载测试集并且构建稀疏矩阵
    for batch in range(4):
        matrix = zeros((5000, 2500), dtype=int8)  # 初始化2W行,2W列的大矩阵
        x_train = []
        y_train = []
        ids = []
        begin_x = batch * 5000  # 初始化矩阵索引
        input_file = open('../data/user_tag_query.2W.TEST')
        for line in islice(input_file, begin_x, None):
            if begin_x >= (batch + 1) * 5000:
                break
            print "预测数据的行数{0}".format(begin_x)
            begin_y = 0
            list = line.split('\t')
            testList = list[1:]
            id = list[0:1]
            ids.append(id)
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
        print "预测批次...{0}".format(batch)
        # print "PCA......{0}".format(batch)
        # pca.fit(matrix)
        # train_data = pca.transform(matrix)  # Fit the model with X and 在X上完成降维.
        # print "pca over~{0}".format(batch)
        # print pca.explained_variance_
        # print train_data.shape
        # np.savetxt('predict_x_data', train_data)

        y_pred = []
        # predict
        print('Saving...{0}'.format(batch))  # 保存预测结果
        count = 0

        for x_test in np.array(matrix):
            thisDoc1 = array(x_test).reshape((1, -1))
            y_predTemp1 = clf1.predict(thisDoc1)
            y_predTemp2 = clf2.predict(thisDoc1)
            y_predTemp3 = clf3.predict(thisDoc1)
            print y_predTemp1
            print y_predTemp2
            print y_predTemp3
            print '---------'
            writer.write(
                str(ids[count][0]) + ' ' + str(y_predTemp1[0]) + ' ' + str(y_predTemp2[0]) + ' ' + str(
                    y_predTemp3[0]) + '\n')
            count += 1

print 'over!#$%^&*()^%$#@!@#$%^&*(~~~~~~~~~~~'

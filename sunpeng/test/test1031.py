# encoding=utf-8
# 预测final版本
import jieba
from numpy import *
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
import codecs

# # 加载本地模型
# clf1 = joblib.load("train_model1.m_1031")
# clf2 = joblib.load("train_model2.m_1031")
# clf3 = joblib.load("train_model3.m_1031")

from sklearn.decomposition import PCA

COMPONENT_NUM = 1000  # 设置pca降维的维度值
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
keywordList = []
# 构建稀疏矩阵的列
fkeywords = open('../data/keywords_filtered.txt', 'r')
for keyword in fkeywords.readlines():
    keywordList.append(str(keyword.strip('\n')))

print(keywordList)  # 大关键词列表

from itertools import islice

with codecs.open('predict.csv_1031', 'w', 'gbk') as writer:
    # 加载测试集并且构建稀疏矩阵
    for batch in range(20):
        matrix = zeros((1000, 20000), dtype=int8)  # 初始化2W行,2W列的大矩阵
        x_train = []
        y_train = []
        ids = []
        begin_x = batch * 1000  # 初始化矩阵索引
        input_file = open('../data/user_tag_query.2W.TEST')
        for line in islice(input_file, begin_x, None):
            if begin_x >= (batch + 1) * 1000:
                break
            print "测试数据的行数{0}".format(begin_x)
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
                    matrix[begin_x - 1000 * batch][begin_y] = 1
                else:
                    begin_y = begin_y + 1
            begin_x = begin_x + 1

        # 降维
        print ids
        print "PCA......{0}".format(batch)
        np.savetxt('matrix-{0}'.format(batch), matrix)
        pca.fit(matrix)
        train_data = pca.transform(matrix)  # Fit the model with X and 在X上完成降维.
        print "pca over~{0}".format(batch)
        print pca.explained_variance_ratio_
        print train_data.shape
        # np.savetxt('predict_x_data', train_data)

        y_pred = []
        # predict
        print('Saving...{0}'.format(batch))  # 保存预测结果
        count = 0

        # for x_test in np.array(train_data):
        #     y_predTemp1 = clf1.predict(x_test)
        #     y_predTemp2 = clf2.predict(x_test)
        #     y_predTemp3 = clf3.predict(x_test)
        #     print y_predTemp1
        #     print y_predTemp2
        #     print y_predTemp3
        #     print '---------'
        #     writer.write(
        #         str(ids[count][0]) + ' ' + str(y_predTemp1[0]) + ' ' + str(y_predTemp2[0]) + ' ' + str(
        #             y_predTemp3[0]) + '\n')
        #     count += 1

print 'over!#$%^&*()^%$#@!@#$%^&*(~~~~~~~~~~~'

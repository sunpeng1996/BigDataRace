# encoding=utf-8
import jieba
from numpy import *
import numpy as np

# 此文件被py2版本执行

fkeywords = open('../data/keywords_filtered_feature_1.txt', 'r')
ftrainingData = open('../data/user_tag_query.2W.TRAIN', 'r')

matrix = zeros((1000, 20000))  # 初始化2W行,2W列的大矩阵

keywordList = []

for keyword in fkeywords.readlines():
    keywordList.append(str(keyword.strip('\n')))

print(keywordList)  # 大关键词列表

x_train = []
y_train = []

begin_x = 0  # 初始化矩阵索引

# 遍历训练集的每一行数据
for line in ftrainingData.readlines():
    print "训练数据的行数{0}".format(begin_x)
    if begin_x >= 1000:
        break
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
            matrix[begin_x][begin_y] = 1
        else:
            begin_y = begin_y + 1
    begin_x = begin_x + 1

print matrix.shape
print np.array(y_train).shape

# 保存产生memory超出
# print "saving traindata......"
# import pickle
#
# output_x = open('data_x.pkl', 'wb')
# output_y = open('data_y.pkl', 'wb')
# pickle.dump(np.array(matrix), output_x)
# pickle.dump(np.array(y_train))

# print "saved..."

# 降维
from sklearn.decomposition import PCA

COMPONENT_NUM = 1000  # 设置pca降维的维度值
# train_data = np.array(matrix)
print "PCA......"
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(matrix)
train_data = pca.transform(matrix)  # Fit the model with X and 在X上完成降维.
print "svm over~"
print train_data.shape

from sklearn.svm import SVC

clf1 = SVC()
clf2 = SVC()
clf3 = SVC()
print "training......"

# for i in range(999):
clf1.fit(train_data, np.array(y_train)[:, 0])  # 训练
clf2.fit(train_data, np.array(y_train)[:, 1])  # 训练
clf3.fit(train_data, np.array(y_train)[:, 2])  # 训练
# print 'training number is {0}'.format(i)

print "train over"

# 保存模型到本地
from sklearn.externals import joblib

joblib.dump(clf1, "train_model1.m")
joblib.dump(clf2, "train_model2.m")
joblib.dump(clf3, "train_model3.m")

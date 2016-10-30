# encoding=utf-8
import jieba
from numpy import *
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

# 加载本地模型
clf1 = joblib.load("train_model1.m")
clf2 = joblib.load("train_model2.m")
clf3 = joblib.load("train_model3.m")

# 加载测试集并且构建稀疏矩阵
fkeywords = open('../data/keywords_filtered.txt', 'r')

matrix = zeros((3000, 20000), dtype=int8)  # 初始化2W行,2W列的大矩阵

keywordList = []

for keyword in fkeywords.readlines():
    keywordList.append(str(keyword.strip('\n')))

print(keywordList)  # 大关键词列表

x_train = []
y_train = []
ids = []
begin_x = 0  # 初始化矩阵索引

with open('../data/user_tag_query.2W.TEST') as f:
    for line in f.readlines():

        if begin_x >= 3000:
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
                matrix[begin_x][begin_y] = 1
            else:
                begin_y = begin_y + 1
        begin_x = begin_x + 1

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


y_pred = []
# predict
for x_test in np.array(matrix):
    y_predTemp1 = clf1.predict(x_test)
    y_predTemp2 = clf2.predict(x_test)
    y_predTemp3 = clf3.predict(x_test)
    print y_predTemp1
    print y_predTemp2
    print y_predTemp3
    print '---------'

# print('Saving...')  # 保存预测结果
# with open('predict.csv') as writer:
#     count = 0
#     for p in y_pred:
#         writer.write(ids[count] + "," + p[0] + "," + p[1] + "," + p[2] + '"\n')
#         count += 1

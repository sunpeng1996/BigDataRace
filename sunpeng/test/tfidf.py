# -*- coding: cp936 -*-
import jieba
import jieba.posseg as pseg
import os
import sys
from sklearn import feature_extraction
import string
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import sys
import numpy as np

reload(sys)
sys.setdefaultencoding("utf-8")
# sys.path.append("C:\Users\Administrator\Desktop\9.17")
from numpy import *

fr = open('../data/filter_zero.csv')
fr_list = fr.read()
dataList = fr_list.split('\n')
dataList = dataList[4:]
data = []
for oneline in dataList:
    data.append(" ".join(jieba.cut(oneline)))

print "��ͷִ�over"

# ���õ��Ĵ���ת��Ϊ��Ƶ����
freWord = CountVectorizer()

print "��Ƶ����over"

# ͳ��ÿ�������tf-idfȨֵ
transformer = TfidfTransformer()
# �����tf-idf(��һ��fit_transform),������ת��Ϊtf-idf����(�ڶ���fit_transformer)
tfidf = transformer.fit_transform(freWord.fit_transform(data))

# print tfidf
print "tfidf over~"
# ��ȡ�ʴ�ģ���е����д���
word = freWord.get_feature_names()

# �õ�Ȩ��
# weight = tfidf.toarray()
weight = np.array(tfidf, dtype='float16')

print "weight over"

tfidfDict = {}
for i in range(len(weight)):
    for j in range(len(word)):
        getWord = word[j]
        getValue = weight[i][j]
        if getValue != 0:
            if tfidfDict.has_key(getWord):
                tfidfDict[getWord] += string.atof(getValue)
            else:
                tfidfDict.update({getWord: getValue})
sorted_tfidf = sorted(tfidfDict.iteritems(),
                      key=lambda d: d[1], reverse=True)

print "����over"
fw = open('result.txt', 'w')
for i in sorted_tfidf:
    fw.write(i[0] + '\t' + str(i[1]) + '\n')

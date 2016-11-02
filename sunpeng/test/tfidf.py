# -*- coding: utf-8 -*-
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
from numpy import *
from itertools import islice

tfidfDict = {}
for batch in range(20):
    ftrainingData = open('../data/filter_zero.csv', 'r')
    begin = batch * 800
    data = []
    for line in islice(ftrainingData, begin, None):
        if begin >= (batch + 1) * 800:
            break
        print len(line)
        data.append(" ".join(jieba.cut_for_search(line[39:])))
        begin = begin + 1
    print "结巴分词over---{0}".format(batch)
    # 将得到的词语转换为词频矩阵
    freWord = CountVectorizer()
    print "词频矩阵over"
    # 统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
    # 计算出tf-idf(第一个fit_transform),并将其转换为tf-idf矩阵(第二个fit_transformer)
    tfidf = transformer.fit_transform(freWord.fit_transform(data))
    print type(tfidf)  # <class 'scipy.sparse.csr.csr_matrix'>
    print tfidf.shape
    print "tfidf over~"
    # 获取词袋模型中的所有词语
    word = freWord.get_feature_names()
    # 得到权重
    weight = tfidf.toarray()
    print type(weight)  # <type 'numpy.ndarray'>
    print weight
    print "weight over"
    print len(weight)
    print weight.shape
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
    fw = open('../tfidf/result__112__{0}.txt'.format(batch), 'w')
    for i in sorted_tfidf:
        fw.write(i[0] + '\t' + str(i[1]) + '\n')

sorted_tfidf = sorted(tfidfDict.iteritems(),
                      key=lambda d: d[1], reverse=True)

print "排序over"
fw = open('../tfidf/result_112.txt', 'w')
for i in sorted_tfidf:
    fw.write(i[0] + '\t' + str(i[1]) + '\n')

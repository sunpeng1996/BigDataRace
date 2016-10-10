# encoding=utf-8
import jieba
from numpy import *
from collections import Counter

import sys

reload(sys)

sys.setdefaultencoding('utf-8')

f = open('../data/user_tag_query.2W.TRAIN', 'r')

# f = open('../data/test.txt', 'r')

wordSet = set()

for line in f.readlines():
    list = line.split('\t')
    print list
    preprocessList = list[4:]
    tempSet = set(preprocessList)
    wordSet = wordSet | tempSet

f.close()
globalSet = set()

c = Counter()

for word in wordSet:
    seg_list = jieba.cut_for_search(str(word))  # 结巴分词,搜索引擎模式
    for word_in in seg_list:
        print word_in
        c[word_in] = c[word_in] + 1
        globalSet.add(word_in)

print len(globalSet)

print c

for (key, value) in c.items():
    print str(key) + ":" + str(value)

dict = sorted(c.iteritems(), key=lambda d: d[1], reverse=True)
print dict

import codecs

file = codecs.open('../data/keywords.txt', 'w')

for (key, value) in dict:
    print str(key) + ":" + str(value)
    file.write(str(key) + "\t" + str(value) + "\n")

file.close()

# matrix = zeros((20000, len(globalSet)))  # 初始化大矩阵,20000行-len(globalSet)列

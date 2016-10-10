# encoding=utf-8
# import jieba
# from numpy import *

# 此文件被py3版本执行

fkeywords = open('../data/keywords_filtered.txt', 'r', encoding='utf-8')
ftrainingData = open('../data/user_tag_query.2W.TRAIN', 'r', encoding='utf-8')

# matrix = zeros((20000, 400000))  # 初始化2W行,40W列的大矩阵

keywordList = []

for keyword in fkeywords.readlines():
    keywordList.append(str(keyword.strip('\n')))

print(keywordList)  # 大关键词列表





# for trainingDataLine in ftrainingData.readlines():
#     trainingDataLineList = trainingDataLine.split('\t')
#     data = trainingDataLineList[4:]
#     seg_list = jieba.cut_for_search(str(data))  # 结巴分词,搜索引擎模式,得到每一行的分词列表

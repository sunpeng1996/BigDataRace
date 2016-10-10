# encoding=utf-8
# 此文件是用py3版本运行的
fKeyWords = open('../data/keywords.txt', 'r', encoding='utf-8')
fStopWords = open('../data/stopword', 'r', encoding='utf-8')

stopwordList = []

for fStopWord in fStopWords.readlines():
    fStopWord = fStopWord.strip('\n')
    stopwordList.append(fStopWord)

print(stopwordList)

import codecs

file = codecs.open('../data/keywords_filtered.txt', 'w', encoding='utf-8')

for keyword in fKeyWords.readlines():
    list = keyword.split('\t')
    # print(list)
    word = list[0]
    if word in stopwordList:
        print(word + "是停用词" + '\n')
    else:
        file.write(word + "\n")

fKeyWords.close()
fStopWords.close()
file.close()

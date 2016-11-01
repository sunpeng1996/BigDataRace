# encoding=utf-8
# 数据预处理，过滤掉含有0的数据
ftrainingData = open('../data/user_tag_query.2W.TRAIN', 'r')

f = open('../data/filter_zero.csv', 'w')
for line in ftrainingData.readlines():
    list = line.split('\t')
    if '0' in list:
        print "error data:{0}".format(list)
    else:
        # print "right data:{0}".format(list)
        f.write(line)

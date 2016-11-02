fr = open('result_112.txt')
fw = open('result_112_filternumber.txt', 'w')
i = 0
for line in fr.readlines():
    list = line.split('\t')
    i = i + 1
    fw.write(str(list[0]) + '\n')
    print i

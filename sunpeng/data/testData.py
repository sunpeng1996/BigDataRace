# encoding=utf-8
f1 = open('keywords_filtered_feature_2.txt')
f2 = open('keywords_filtered_feature_1.txt')

a = f1.read()
lista = a.split('\n')

b = f2.read()
listb = b.split('\n')

print len(lista)
print len(listb)

c = list(set(lista).intersection(set(listb)))

print len(c)

import numpy as np
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier()

datax = np.array([[1, 5], [-1, 5], [5, 1], [5, -1], [1, -5], [-1, -5], [-5, 1], [-5, -1]])
datay = np.array(['1', '1', '2', '2', '3', '3', '4', '4'])

clf.fit(datax, datay)

Z = clf.predict(np.array([[4, 3]]))
print Z

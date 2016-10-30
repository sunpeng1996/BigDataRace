# coding = utf-8
import pickle

data_x = open("data_x.pkl", 'rb')
data_y = open("data_y.pkl", 'rb')

dataX = pickle.load(data_x)
dataY = pickle.load(data_y)



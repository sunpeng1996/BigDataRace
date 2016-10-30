# coding = utf-8
import pickle

from itertools import islice

input_file = open("test")
for line in islice(input_file, 2, None):
    print line



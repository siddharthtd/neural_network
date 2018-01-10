import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import datasets as ds


digits = ds.load_digits()
digits.keys()

digits['target']

rix = [x for x in range(len(digits['target']))]
random.shuffle(rix)
rix[:10]


dlen = len(rix)
tlimit = int(dlen*0.75)
train_data = digits['data'][0:tlimit]
train_target = digits['target'][0:tlimit]
test_data = digits['data'][tlimit:]
test_target = digits['target'][tlimit:]
print (len(train_data), len(test_data))


from sklearn import svm

classy = svm.SVC(gamma=0.001, C = 100.)

classy.fit(train_data, train_target)

pred_target = classy.predict(test_data)

def arediff(ix):
    if pred_target[ix]==test_target[ix]:
        return 0
    return 1

diffs = [arediff(x) for x in range(len(test_target))]

print (sum(diffs))

print ((len(diffs)-sum(diffs))/len(diffs))


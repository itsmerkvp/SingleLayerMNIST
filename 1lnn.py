import time
import math
import random
from mnist import MNIST
import numpy as np
from ieee754 import IEEE754

f = open("weights.bin","w")

mndata = MNIST('./data/','rounded_binarized','numpy')
traini, trainl = mndata.load_training()
testi, testl = mndata.load_testing()

weight = np.random.rand(10,784)
index = 0
count = 0
start_time = time.time()

for w in range(60000):
    output = np.zeros(10)
    target = np.zeros(10)
    error  = np.zeros(10)
    target[trainl[w]] = 1

    for i in range(10):
        output[i] = np.matmul(weight[i],traini[w])
        output[i] /= 784
        error[i] = target[i] - output[i]
        for j in range(784):
            weight[i][j] = weight[i][j] + 0.05*error[i]*traini[w][j]

    index = output.argmax()
    if(trainl[w] == index):
        count += 1

print("training accuracy is",(count/600))
print("--- %s seconds ---" % (time.time() - start_time))

for i in range(10):
    for j in range(784):
        a = IEEE754(weight[i][j],1)
        f.write(str(a))
        f.write("\n")

start_time = time.time()
count = 0
for w in range(10000):

    output = np.zeros(10)

    for i in range(10):
        output[i] = np.matmul(weight[i],testi[w])
    
    index = output.argmax()
    if(testl[w] == index):
        count += 1

print("testing accuracy is",(count/100))
print("--- %s seconds ---" % (time.time() - start_time))


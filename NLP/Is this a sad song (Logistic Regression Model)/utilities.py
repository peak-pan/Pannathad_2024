
import numpy as np
import re
from thai_tokenizer import Tokenizer

def sigmoid(z):
    return 1/(1+np.exp(-z))

def Process(sentence):
    tokenizer = Tokenizer()
    i= re.sub(r'[(,)]', '', sentence)
    x = tokenizer(i)
    arr = x.split(' ')
    return arr 

def build_freq(y,x):
    freqs = {}
    print(zip(y,x))
    for yi,xi in zip(y,x):
        for word in Process(xi):
            pair = (word, yi[0])
            if pair in freqs:
                freqs[pair] +=1
            else:
                freqs[pair] = 1
    return freqs


def gradientDescent(x, y, theta, alpha, num_iters):
    m = x.shape[0]
    
    for i in range(0, num_iters):
        z = np.dot(x, theta)
        h = sigmoid(z)
        J = -1/m * (np.dot(y.transpose(), np.log(h)) + np.dot((1-y).transpose(),np.log(1-h)))
        theta += -(alpha/m)*(np.dot(x.transpose(),(h-y)))
        print("epoch:", i," Cost:", J)
    J = float(J)
    return J, theta

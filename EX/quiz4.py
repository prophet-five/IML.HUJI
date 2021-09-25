import numpy as np
import math


def fu():
    s = [-1, 1, 1, 1, 1]
    D = [0.05782, 0.03686, 0.05289, 0.22113, 0.63131]
    w = [1, 1, -1, 1, 1]

    et = 0

    for i in range(len(w)):
        if s[i] != w[i]:
            et += D[i]

    wt = 0.5 * (math.log(1 / et - 1))

    sum = 0

    top = 0.05289 * math.exp(-s[2] * w[2] * wt)

    for i in range(5):
        sum += D[i] * math.exp(-s[i] * w[i] * wt)

    print(top / sum)

def fu2():
    import numpy as np
    # replace with your D
    D = np.array([0.05782, 0.03686, 0.05289, 0.22113, 0.63131])
    # replace with your response vector
    y = np.array([-1, 1, 1, 1, 1])
    # replace with your h
    h = np.array([1, 1, -1, 1, 1])
    epsilon_idx = [i for i in range(len(D)) if y[i] != h[i]]
    epsilon = np.sum(D[epsilon_idx])
    w = 0.5 * np.log((1 / epsilon) - 1)
    D = np.multiply(D, np.exp(- w * np.multiply(y, h)))
    D = D / np.sum(D)
    # take your right entry
    print(D)

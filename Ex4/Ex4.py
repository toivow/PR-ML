# Toivo Wuoti, H281977

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np


X = np.loadtxt("X.dat", unpack=True)
y = np.loadtxt("y.dat", unpack=True)

X = X - np.array([np.mean(X, axis=1)]).T
for i in range(len(y)):
    if y[i] == -1:
        y[i] = 0

def accuracy(predict, gt):

    correct = 0
    for i in range(0, len(predict)):
        if predict[i] == gt[i]:
            correct += 1

    return correct/len(predict) * 100

def normalize(predict):

    for i in range(0, len(predict)):
        if predict[i] > 0:
            predict[i] = 1
        else:
            predict[i] = 0

    return predict

def logsig(w):

    return 1/(1 + np.exp(-w))

# # Slice to training data and test data
# slice = None
#
# X_train = X[0:slice]
# y_train = y[0:slice]
#
# X_test = X[slice:]
# y_test = y[slice:]

loop_end = 100

reg_scores = []
reg_weights = np.zeros((loop_end, 2))

sse_scores = []
sse_weights = np.zeros( (loop_end+1, 2) )

ml_scores = []
ml_weights = np.zeros( (loop_end+1, 2) )

mu = 0.001

sse_weights[0] = [1, -1]
ml_weights[0] = [1, -1]

class1_idxs = np.where(y == 1)
class0_idxs = np.where(y == 0)


for i in range(loop_end):

    # SKlearn linear regression
    reg = LogisticRegression(penalty='none', fit_intercept=False).fit(X.T, y.T)
    reg_scores.append(accuracy(reg.predict(X.T), y.T))
    reg_weights[i] = reg.coef_

    w_sse = np.array([sse_weights[i]])[0]

    y_hat = logsig(w_sse @ X)

    # SSE prediction
    y_sse = normalize(2*logsig(w_sse @ X) - 1)
    sse_scores.append(accuracy(y_sse, y))

    # SSE weights
    sse_weights[i+1] = w_sse - mu*(((-2*(y - y_hat) * (y_hat)) * ((1 - (y_hat))).T @ X.T))

    w_ml = np.array([ml_weights[i]])[0]

    u = logsig(w_ml @ np.squeeze(X[:,class0_idxs]))
    gradient0 = (-u) @ np.squeeze(X[:,class0_idxs]).T

    v = logsig(w_ml @ np.squeeze(X[:,class1_idxs]))
    gradient1 = (1 - v) @ np.squeeze(X[:,class1_idxs]).T

    # ML prediction
    y_ml = normalize(2*logsig(w_ml @ X) - 1)
    ml_scores.append(accuracy(y_ml, y))

    # ML weights
    ml_weights[i+1] = w_ml + mu* (gradient1 + gradient0)


fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(reg_weights[:,0], reg_weights[:,1], "kx")
ax1.plot(sse_weights[:,0], sse_weights[:,1], "co-", markersize=4)
ax1.plot(ml_weights[:,0], ml_weights[:,1], "ro-", markersize=4)
ax1.set_xlabel("w1")
ax1.set_ylabel("w2")

ax2.plot(reg_scores, "k--")
ax2.plot(sse_scores, "c-")
ax2.plot(ml_scores, "r-")
ax2.legend(["SKlearn", "SSE", "ML"])
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Accuracy / %")

plt.show()

"""Answers to question 3:

The plots are similar, however they're not entirely the same. ML seems to take larger leaps in the beginning, which 
leads to it getting a higher likelihood faster. SSE seems to slow down its approach to the end of the loop, which is why
it doesn't seem to achieve the point x, however the accuracy is still good."""
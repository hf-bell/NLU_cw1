import sys
import time
import numpy as np

from utils import *
from rnnmath import *
from sys import stdout

vocab_size = 3
hidden_dims = 2
out_vocab_size = 3

U = np.random.randn(hidden_dims, hidden_dims)*np.sqrt(0.1)
V = np.random.randn(hidden_dims, vocab_size)*np.sqrt(0.1)
W = np.random.randn(out_vocab_size, hidden_dims)*np.sqrt(0.1)


V[0][0]=0.7
V[0][1]=0.3
V[0][2]=0.4
V[1][0]=0.6
V[1][1]=0.9
V[1][2]=0.7

W[0][0]=0.6
W[0][1]=0.5
W[1][0]=0.2
W[1][1]=0.6
W[2][0]=0.4
W[2][1]=0.2

U[0][0]=0.9

U[0][1]=0.8
U[1][0]=0.5
U[1][1]=0.3


def predict(x):
    s = np.zeros((len(x) + 1, hidden_dims))
    y = np.zeros((len(x), out_vocab_size))

    for t in range(0,len(x)):
        x_1h = make_onehot(x[t], 3)
        test = np.dot(U, s[(t-1)])
        net_in = np.dot(V, x_1h) + np.dot(U, s[(t-1)])

        
        s[t] = sigmoid(net_in)
        net_out = np.dot(W, s[t])
        y[t] = softmax(net_out)
    return y,s

##def compute_loss(x, d):
##    loss = 0.
##    y_pred, hiddens = predict(x)
##    #pred = np.where(y_pred == numpy.amax(arr))
##    #pred = np.amax(y_pred, axis = 1)
##    
##    pred = np.argmax(y_pred, axis = 1)
##    #d_1h = make_onehot(d, 3)
##    #print(y_pred)
##    print('Target: {} vs pred: {}'.format(d, pred))
##    loss = -np.sum(d * np.log(pred + 1e-9))
##    return loss


def compute_loss(x, d):
    loss = 0.
    y_pred, hiddens = predict(x)
    for t_index, t in enumerate(d):
        d_t_1h = make_onehot(t, 3)
        t_loss = 0.
        print('Target: {} vs pred: {}'.format(d_t_1h, y_pred[int(t_index)]))
        for i in d_t_1h:
            log_y_hat__j_t = np.log(y_pred[int(t_index)][int(i)])
            i_loss = log_y_hat__j_t*i
            t_loss += i_loss
            t_loss = t_loss * -1
            loss += t_loss
        return loss

def compute_mean_loss(X, D):
    mean_loss = 0.
    N = len(D)

    for i in range(len(X)):
        mean_loss += compute_loss(X[i], D[i])

        mean_loss = mean_loss/N

    return mean_loss


y_exp = np.array([[ 0.39411072,  0.32179748,  0.2840918 ], [ 0.4075143,   0.32013043,  0.27235527], [ 0.41091755,  0.31606385,  0.2730186 ], [ 0.41098376,  0.31825833,  0.27075792], [ 0.41118931,  0.31812307,  0.27068762], [ 0.41356637,  0.31280332,  0.27363031], [ 0.41157736,  0.31584609,  0.27257655]])
s_exp = np.array([[ 0.66818777,  0.64565631], [ 0.80500806,  0.80655686], [ 0.85442692,  0.79322425], [ 0.84599959,  0.8270955 ], [ 0.84852462,  0.82794442], [ 0.89340731,  0.7811953 ], [ 0.86164528,  0.79916155], [ 0., 0.]])
U_exp = np.array([[ 0.89990596,  0.79983619], [ 0.5000714,   0.30009787]])
V_exp = np.array([[ 0.69787081,  0.30129314,  0.39888647], [ 0.60201076,  0.89866058,  0.70149262]])
W_exp = np.array([[ 0.57779081,  0.47890397], [ 0.22552931,  0.62294835], [ 0.39667988 , 0.19814768]])


x = np.array([0,1,2,1,1,0,2])
d = np.array([1,2,1,1,1,1,1])
x2 = np.array([1,1,0])
d2 = np.array([1,0,2])
x3 = np.array([1,1,2,1,2])
d3 = np.array([1,2,1,2,1])
loss_expected = 8.19118156763
loss2_expected = 3.29724981191
loss3_expected = 6.01420605985
mean_loss_expected = 1.16684249596

y,s = predict(x)
if not np.isclose(y_exp, y, rtol=1e-08, atol=1e-08).all():
	print("y expected\n{0}".format(y_exp))
	print("y received\n{0}".format(y))
else:
	print("y passed")
if not np.isclose(s_exp, s, rtol=1e-08, atol=1e-08).all():
	print("\ns expected\n{0}".format(s_exp))
	print("s received\n{0}".format(s))
else:
	print("s passed")


print("\n### computing loss and mean loss")
loss = compute_loss(x,d)
loss2 = compute_loss(x2,d2)
loss3 = compute_loss(x3,d3)
mean_loss = compute_mean_loss([x,x2,x3],[d,d2,d3])
if not np.isclose(loss_expected, loss, rtol=1e-08, atol=1e-08) or not np.isclose(loss2_expected, loss2, rtol=1e-08, atol=1e-08) or not np.isclose(loss3_expected, loss3, rtol=1e-08, atol=1e-08):
	print("loss expected: {0}".format(loss_expected))
	print("loss received: {0}".format(loss))
	print("loss2 expected: {0}".format(loss2_expected))
	print("loss2 received: {0}".format(loss2))
	print("loss3 expected: {0}".format(loss3_expected))
	print("loss3 received: {0}".format(loss3))
else:
	print("loss passed")
if not np.isclose(mean_loss_expected, mean_loss, rtol=1e-08, atol=1e-08):
	print("mean loss expected: {0}".format(mean_loss_expected))
	print("mean loss received: {0}".format(mean_loss))
else:
	print("mean loss passed")



import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

def cross_entropy_error(y, t):
    # delta = 1e-7
    # return -np.sum(t*np.log(y + delta))
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + 1e-7)) / batch_size
    # return -np.sum(np.log(y[np.arrange(batch_size), t] + 1e-7)) / batch_size  # one hot encoding이 아닐 경우

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)


    for idx in range(x.size):
        # f(x+h) 
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad
    return x


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label = True)

print(x_train.shape)
print(t_train.shape)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
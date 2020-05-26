# Anat Sinay 312578149, Harel Doitch 203249842
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

def load_DB():
    mnist = fetch_openml('mnist_784')
    X = mnist["data"].astype('float64')
    y = mnist["target"].astype(np.int)
    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))
    X = np.c_[X, np.ones(X.shape[0])]  # add one as last element
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # split to train sets and test sets
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test


def softmax(alpha):
    exp_alpha = np.exp(alpha - np.max(alpha))       # - max for preventing overflow
    sum_exp_alpha = np.sum(exp_alpha, axis=1)
    return (exp_alpha.T/sum_exp_alpha).T

def calcAccuracy(x, w, t):
    y = softmax(x.dot(w.T))
    prediction = np.sum(np.argmax(y, axis=1) == t)
    return 100.0 * prediction / y.shape[0]


def gradientDescent(w, x, t, etha):
    sm = softmax(x.dot(w.T))
    grad = ((sm - t).T).dot(x)
    return w - etha * grad

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_DB()

    N = 10
    sigma_i = 10 ** -3
    etha = 0.0001
    min_delta_accuracy = 10e-3
    delta_accuracy = 100
    accuracy = 0

    t_train = np.identity(N)[y_train]  # one hot
    t_test = np.identity(N)[y_test]  # one hot
    W = np.random.normal(0, sigma_i, [10, X_train.shape[1]])

    while delta_accuracy >= min_delta_accuracy:
        old_accuracy = accuracy

        W = gradientDescent(W, X_train, t_train, etha)
        accuracy = calcAccuracy(X_test, W, y_test)
        delta_accuracy = abs(old_accuracy - accuracy)

    print('accuracy =' + str(accuracy))
import numpy as np
import matplotlib.pyplot as plt

delta = 0.1
alpha = 0.01
data = np.genfromtxt('C:\Users\harel\Desktop\data.csv',delimiter=',')
X = np.ones((data.shape[0] - 1, data.shape[1] - 1))
X[:,:] = data[1:,:-1]
m = X.shape[0]

# scaling X
mean = np.mean(X, 0)
var = np.std(X, 0)

X = (X-mean) / var
Y = data[1:,data.shape[1] - 1]
Teta = np.ones(data.shape[1] - 1)

errors = []
error = (1 / (2 * m)) * (np.sum(np.square(X.dot(Teta) - Y)))
errors.append(error)
iterationCounter = 0
while error < delta:
	# gradient descent
	a = np.sum(X.dot(Teta) - Y)
	b = X.T.dot(a)
	
	Teta = Teta - alpha * X.T.dot(X.dot(Teta) - Y) / m
	iterationCounter += 1
	error = (1 / (2 * m)) * (np.sum(np.square(X.dot(Teta) - Y)))
	errors.append(error)

plt.plot(errors)
plt.ylabel('Error')
plt.show()
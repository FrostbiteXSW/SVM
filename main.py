import abc
import random
import matplotlib.pyplot as plt
import numpy as np
from time import time
from support_vector_classifier import SupportVectorClassifier
from polynomial import PolynomialKernel

random.seed(time())

b = 1/10
cnt = 20

if __name__ == '__main__':
	model = SupportVectorClassifier(kernel=PolynomialKernel(degree=1))

	data = []
	label = []

	# below y = x - b
	for i in range(cnt):
		x1 = random.random() * (1 - b) + b
		x2 = random.random() * (x1 - b)
		data.append([x1, x2])
		label.append(-1)

	# above y = x + b
	for i in range(cnt):
		x1 = random.random() * (1 - b)
		x2 = random.random() * (1 - b - x1) + x1 + b
		data.append([x1, x2])
		label.append(1)

	data = np.array(data)
	label = np.array(label)
	# x = np.linspace(0, 1, 10)
	# y = x - b
	# plt.plot(x, y, color='red', label='y = x - 1/4')
	plt.scatter(data[0:cnt, 0], data[0:cnt, 1], color='green')
	# y = x + b
	# plt.plot(x, y, color='blue', label='y = x + 1/4')
	plt.scatter(data[cnt:cnt*2, 0], data[cnt:cnt*2, 1], color='blue')

	model.fit(data, label)
	plt.scatter(model.X[:, 0], model.X[:, 1], color='red')
	x0, x1 = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
	x = np.array([x0, x1]).reshape(2, -1).T
	cp = plt.contour(x0, x1, model.distance(x).reshape(100, 100), np.array([-1, 0, 1]),
	                 colors='black', linestyles=('dashed', 'solid', 'dashed'))
	plt.clabel(cp, fmt='y=%.f', inline=True, fontsize=20)
	plt.xlim(0, 1)
	plt.ylim(0, 1)
	plt.gca().set_aspect('equal', adjustable='box')
	# plt.legend()
	plt.show()

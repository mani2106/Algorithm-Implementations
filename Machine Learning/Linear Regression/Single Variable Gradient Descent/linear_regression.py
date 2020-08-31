""" Implements Linear regression with Gradient descent """

import gradient_descent as gd
import plot_data as plot

import numpy as np
from sklearn.datasets import make_regression

X, y, expected_theta = make_regression(n_samples=45, n_features=1, n_targets=1, coef=True, noise=0.1)


alpha = 0.02

# Randomly initialize theta
init_theta = np.random.normal()

hist, theta = gd.gradient_descent(X.ravel(), y, init_theta, alpha, 200)

print('Calculated theta: ', theta, 'Expected theta: ', expected_theta)

# Plot static image
# plot.plot_data(hist, y, theta)

# Generating gif
plot.generate_plot_gif(hist, y)
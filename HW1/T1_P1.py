#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np
import matplotlib.pyplot as plt

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

############# Parts 3 and 4 #############

def kernel(x, xp, tau):
    return np.exp(-(np.linalg.norm(x - xp) ** 2) / tau)

def compute_loss(tau):
    loss = 0.
    for x, y in data:
        f = 0.
        for xn, yn in data:
            if x != xn:
                f += kernel(x, xn, tau) * yn
        loss += (y - f) ** 2
    return loss

taus = [0.01, 2., 100.]
x = np.arange(0, 12, .1)

def prediction(x, tau):
    return sum([kernel(xn, x, tau) * yn for xn, yn in data])

for tau in taus:
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))
    y = [prediction(xi, tau) for xi in x]
    plt.plot(x, y, label=f'\\tau = {tau}')

plt.xlabel('x*')
plt.ylabel('f(x*)')
plt.title('Effect of lengthscale \\tau on predictions')
plt.legend(loc='upper right')
plt.savefig('lengthscales.png')
plt.show()


############# Part 5 (Bonus) #############

def gradient(W, tau):
    diff = 0.
    for i, (x, y) in enumerate(W):
        f = 0.
        for j, (xn, yn) in enumerate(W):
            if i != j:
                f += np.exp(-(np.linalg.norm(xn - x) ** 2) / tau) * yn
        f = y - f
        g = 0.
        for j, (xn, yn) in enumerate(W):
            if i != j:
                a = (np.linalg.norm(xn - x) / tau) ** 2
                b = np.exp(-(np.linalg.norm(xn - x) ** 2) / tau)
                g += a * b * yn
        diff += f * -g
    return diff

# Referenced: https://www.geeksforgeeks.org/how-to-implement-a-gradient-descent-in-python-to-find-a-local-minimum/
def gradient_descent(tau, iterations = 10000, learning_rate = .1, stop_threshold = 1e-6):
    previous_cost = None
    for _ in range(iterations):
        current_cost = compute_loss(tau)
        if previous_cost and abs(previous_cost - current_cost) < stop_threshold:
            break
        previous_cost = current_cost
        tau = tau - learning_rate * gradient(data, tau)
    return tau

opt_tau = gradient_descent(2.)
print("Loss for optimal tau = " + str(opt_tau) + ": " + str(compute_loss(opt_tau)))
y = [prediction(xi, opt_tau) for xi in x]
plt.figure(2)
plt.clf()
plt.plot(x, y, label=f'\\tau = {opt_tau}')
plt.xlabel('x*')
plt.ylabel('f(x*)')
plt.title('Effect of optimal lengthscale \\tau (from gradient descent) on predictions')
plt.legend(loc='upper right')
plt.savefig('lengthscale-optimal.png')
plt.show()

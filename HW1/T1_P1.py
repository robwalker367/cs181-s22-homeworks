#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

import numpy as np

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

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

for tau in (0.5, 1., 50.):
    print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))
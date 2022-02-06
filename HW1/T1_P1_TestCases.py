import numpy as np
from T1_P1 import compute_loss

'''
Instructions for using this Autograder:
1. Make sure it is in the same immediate directory/folder as your implementation file, which *must* be called T1_P1.py
2. Run this only after you have implemented the function compute_loss, which returns the loss for a tau.
3. The test cases this Autograder uses are distinct from the tau values specified in the homework.
'''
data = [(0., 0.),
        (1., 0.5),
        (2., 1),
        (3., 2),
        (4., 1),
        (6., 1.5),
        (8., 0.5)]


tau1 = 0.5
tau2 = 1
tau3 = 50

testloss1 = 6.593154963000763
testloss2 = 4.0672962718889614
testloss3 = 92.99484439811006

case1_checker = np.abs(testloss1 - compute_loss(tau1)) < 0.001
if case1_checker:
    case1_checker = "Pass"
else:
    case1_checker = "Fail"
case2_checker = np.abs(testloss2 - compute_loss(tau2)) < 0.001
if case2_checker:
    case2_checker = "Pass"
else:
    case2_checker = "Fail"
case3_checker = np.abs(testloss3 - compute_loss(tau3)) < 0.01
if case3_checker:
    case3_checker = "Pass"
else:
    case3_checker = "Fail"

print("Your test case results are : ", case1_checker, case2_checker, case3_checker)
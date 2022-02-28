# -*- coding: utf-8 -*-
"""CS181HW0_v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fj23jTQyrxXG_meDsEuWdkJuMqP1b9JI

CS181

### Problem 14
"""

import numpy 
import numpy.random 
import csv

# 1)
numpy.random.seed(181)
N=20
points=[(numpy.random.uniform(-10,10),numpy.random.uniform(20,80)) for i in range(N)]

# 2)

x=[points[i][0] for i in range(len(points))]
y=[points[i][1] for i in range(len(points))]

with open('points.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the data
    writer.writerows(numpy.array([x,y]).T)

data=[]
with open('points.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='\'')
    for row in reader:
        data.append((float(row[0]),float(row[1])))


# Optional 

# 3)
print('Question 3')
def f(x,y):
    return ((y+10)*x)/5
z=[f(x,y) for (x,y) in points]
print('The mean and std are {} and {} respectively.'.format(numpy.mean(z),numpy.std(z)))


# 4)
print('Question 4')
maximum=max([y for (x,y) in points])
ans_4=[(x,y) for (x,y) in points if y==maximum]
if len(ans_4)==1:
    print('The data point (x,y) with the largest y value is {}'.format(ans_4[0]))
else:
    print('The data points (x,y) with the largest y value are {}'.format(ans_4))
    
# 5)
print('Question 5')
ans_5=sum([y for (x,y) in points if x>0])
print('The sum of y-values of all points with positive x-value is {}'.format(ans_5))

"""### Problem 15"""

import numpy as np

# 1)
print('Question 1')
ans_1=np.arange(10)
print(ans_1)

# 2)
print('Question 2')
ans_2=ans_1.reshape((2,5))
print(ans_2)

# 3)
print('Question 3')
ans_3=np.vstack((ans_2, np.arange(10,15)))
print(ans_3)

# 4)
print('Question 4')
ans_4=np.hstack((ans_3, np.ones(3).reshape(3,1)))
print(ans_4)

# 5)
print('Question 5')
vec=[0,1,0,0,0,0]
# Picks up the second column of ans_4
ans_5=np.dot(ans_4,vec)
print(ans_5)

# 6)
print('Question 6')
a,b=ans_4.shape
print("Number of even elements:",a*b-sum(sum(ans_4%2)))
print("Solution to 6):",sum([sum([j for j in i if j%2==0]) for i in ans_4]))
# Using a loop
count=0
for i in ans_4:
    for j in i:
        if j%2==0:
            count+=j


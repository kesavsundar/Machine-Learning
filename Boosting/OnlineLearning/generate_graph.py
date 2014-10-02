__author__ = 'kesavsundar'
import matplotlib.pyplot as plt
import numpy as np


with open("random_output_file", 'rb') as random_output:
    random = [line.strip() for line in random_output]
random_x = list()
random_y = list()
for row in random:
    val = row.split('::')
    if val[0] != '0':
        random_x.append(val[0])
        random_y.append(val[1])

with open("min_output_file", 'rb') as min_output:
    min = [line.strip() for line in min_output]

min_x = list()
min_y = list()
for row in min:
    val = row.split('::')
    if val[0] != '0':
        min_x.append(val[0])
        min_y.append(val[1])



plt.plot(random_x, random_y)
plt.plot(min_x, min_y)
plt.xlabel("% of Data - Increased by 5% each time")
plt.ylabel("Accuracy")

plt.legend(['random', 'min'], loc='upper left')

plt.show()
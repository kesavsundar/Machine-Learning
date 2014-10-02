import NeuralNetwork as net
import numpy as np
# example II
x = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 1]])
net_structure = np.array([8, 3, 8])
neuralnet = net.NeuralNetwork(x, x, net_structure)
neuralnet.encode()
neuralnet.decode()

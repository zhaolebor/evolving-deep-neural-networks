
# coding: utf-8

# # CoDeepNEAT demo
# ## CS081 project checkpoint demo
# ### Harsha Uppli, Alan Zhao, Gabriel Meyer-Lee
# 
# The following notebook demonstrates using CoDeepNEAT to solve CIFAR-10

# In[1]:


from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from math import pi, floor
from random import random
from codeepneat import codeepneat, config, population, chromosome, genome, visualize
import pickle
import numpy as np
import keras
import sys

def fitness(network, data):
    network.fit(data[0], data[1],  epochs=8)
    loss, acc = network.evaluate(data[2], data[3])
    return acc

def main():
  (x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()
  y_train_all = y_train_all[:,0]
  y_test = y_test[:,0]
  
  x_train_all = np.reshape(x_train_all, (x_train_all.shape[0], 32, 32, 3)).astype('float32') / 255
  x_test = np.reshape(x_test, (x_test.shape[0], 32, 32, 3)).astype('float32') / 255
  y_train_all = keras.utils.np_utils.to_categorical(y_train_all)
  y_test = keras.utils.np_utils.to_categorical(y_test)
  
  data = [x_train_all, y_train_all, x_test, y_test]
  eval_best(sys.argv[1], data)



def eval_best(model_file, data):
    config.load('configCIFAR10')
    model = keras.models.load_model(model_file)
    model.fit(data[0], data[1], epochs=50)
    loss, fitness = model.evaluate(data[2], data[3])
    print("fitness", fitness)

if __name__=='__main__':
  main()

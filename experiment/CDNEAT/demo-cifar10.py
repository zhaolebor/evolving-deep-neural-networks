
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


# ### Problem: CIFAR10 data set
# 
# Conveniently, it's also built into Keras, which our CoDeepNEAT imiplementation is built off of.

# In[2]:


(x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()
y_train_all = y_train_all[:,0]
y_test = y_test[:,0]

x_train_all = np.reshape(x_train_all, (x_train_all.shape[0], 32, 32, 3)).astype('float32') / 255
x_test = np.reshape(x_test, (x_test.shape[0], 32, 32, 3)).astype('float32') / 255
y_train_all = keras.utils.np_utils.to_categorical(y_train_all)
y_test = keras.utils.np_utils.to_categorical(y_test)

index = np.array(range(len(x_train_all)))
np.random.shuffle(index)
index_train = index[:42500]
index_val = index[42500:]
x_train = x_train_all[index_train]
y_train = y_train_all[index_train]
x_val = x_train_all[index_val]
y_val = y_train_all[index_val]

data = [x_train, y_train, x_val, y_val, x_test, y_test]
print("data shapes")
print("  x train:", x_train.shape)
print("  y train:", y_train.shape)

print("  x val:", x_val.shape)
print("  y val:", y_val.shape)

print("  x test:", x_test.shape)
print("  y test:", y_test.shape)


# ### Configuring NEAT
# 
# Many of the options and inputs are still handled through the config file. The config file has been shortened considerably as many parameters have been eliminated, although many parameters have also been introduced which could be added.

# In[7]:

# ### Fitness
# 
# For this demonstration we'll be using supervised learning to train the networks produced by CoDeepNEAT on CIFAR-10 and will use their accuracy after 5 epochs as our fitness. CIFAR-10, like MNIST, is a 10 category classification problem.

# In[8]:


def fitness(network, data):
    network.fit(data[0], data[1],  epochs=8)
    loss, acc = network.evaluate(data[2], data[3])
    return acc


# ### Evolution
# 
# Evolution with CoDeepNEAT is slightly different than evolution with NEAT. The main difference is coevolution, where we have two separate populations with a hierarchical relationship evolving together.

# In[9]:


def evolve(n, debugging=False):
    if(debugging):
        debug = open("debug.txt", "w")
    else:
        debug = None
    config.load('configCIFAR10')
    # Create 2 separate populations (size is now defined explicitly, but config file can still be used)
    module_pop = population.Population(15, chromosome.ModuleChromo, debug=debug)
    # As the top hierarchical level, the blueprint population needs to be able to see the module population
    blueprint_pop = population.Population(10, chromosome.BlueprintChromo, module_pop, debug=debug)
    # Most of the actual evolving is now handled outside of the population, by CoDeepNEAT
    # Instead of requiring the user to overwrite the evaluation function, CoDeepNEAT evaluates the populations itself,
    # it simply requires a fitness function for the networks it creates passed in as an argument.
    codeepneat.epoch(n, blueprint_pop, module_pop, 25, fitness, data, save_best=True, name='CIFAR10', debug=debug)
    # It will still stop if fitness surpasses the max_fitness_threshold in config file
    # Plots the evolution of the best/average fitness
    visualize.plot_stats(module_pop.stats, name="CIFAR10mod_")
    visualize.plot_stats(blueprint_pop.stats, name="CIFAR10bp_")
    # Visualizes speciation
    #visualize.plot_species(module_pop.species_log, name="NMISTmod_")
    #visualize.plot_species(blueprint_pop.species_log, name="NMISTbp_")


# In[10]:


evolve(25, True)


# In[ ]:


def eval_best(model_file):
    config.load('configCIFAR10')
    model = keras.models.load_model(model_file)
    visualize.draw_net(model, "_" + model_file)    
    model.fit(x_train_all, y_train_all, epochs=50)
    loss, fitness = model.evaluate(x_test, y_test)
    print("fitness", fitness)


# In[ ]:


# eval_best("CIFAR10_best_model_0")


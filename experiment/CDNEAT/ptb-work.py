
# coding: utf-8

# # CoDeepNEAT demo
# ## CS081 project checkpoint demo
# ### Harsha Uppli, Alan Zhao, Gabriel Meyer-Lee
#
# The following notebook demonstrates using CoDeepNEAT to solve the Penn Tree Bank

# In[7]:


from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras import preprocessing
from keras import backend as K
from math import pi, floor
from random import random
from codeepneat import codeepneat, config, population, chromosome, genome, visualize
import pickle
import numpy as np
import keras
import collections


# In[8]:


max_words = 10000
print('Loading data...')
(x_train_all, y_train_all), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)
print(len(x_train_all), 'train sequences')
print(len(x_test), 'test sequences')

num_classes = np.max(y_train_all) + 1
print(num_classes, 'classes')



x_train_all = preprocessing.sequence.pad_sequences(x_train_all, maxlen=30)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=30)[:2240]
y_train_all = keras.utils.to_categorical(y_train_all, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)[:2240]

# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)


print('Convert class vector to binary class matrix '
      '(for use with categorical_crossentropy)')
#Reshaping the input data
# x_train = np.reshape(x_train, (8982, 10000, 1))
# x_test = np.reshape(x_test, (2246, 10000, 1))
# print('y_train shape:', y_train_all.shape)
# print('y_test shape:', y_test.shape)

index = np.array(range(len(x_train_all)))
np.random.shuffle(index)
index_train = index[:7862]
index_val = index[7862:]
x_train = x_train_all[index_train][:7840]
y_train = y_train_all[index_train][:7840]
x_val = x_train_all[index_val]
y_val = y_train_all[index_val]
print('x_train shape', x_train.shape)
print('x_val shape', x_val.shape)
print('x_test shape', x_test.shape)

data = [x_train, y_train, x_val, y_val, x_test, y_test]


# In[9]:


get_ipython().run_cell_magic('file', 'configReuters', '#--- parameters for the robot experiment ---#\n[phenotype]\ninput_nodes         = 30\noutput_nodes        = 46\nconv                = False\nLSTM                = True\n\n[genetic]\nmax_fitness_threshold = 1\n\n# Human reasoning\npop_size              = 10\nprob_addconv          = 0.0\nprob_addLSTM          = 0.0\nprob_addlayer         = 0.1\nprob_mutatelayer      = 0.4\nprob_addmodule        = 0.05\nprob_switchmodule     = 0.3\nelitism               = 1\n\n[genotype compatibility]\ncompatibility_threshold = 3.0\ncompatibility_change    = 0.0\nexcess_coefficient      = 5.0\ndisjoint_coefficient    = 3.0\nconnection_coefficient  = 0.4\nsize_coefficient        = 0.8\n\n[species]\nspecies_size        = 10\nsurvival_threshold  = 0.2\nold_threshold       = 30\nyouth_threshold     = 10\nold_penalty         = 0.2\nyouth_boost         = 1.2\nmax_stagnation      = 15')


# In[10]:


def fitness(network, data):
    network.fit(data[0], data[1],  epochs=5, batch_size=32)
    loss, acc = network.evaluate(data[2], data[3], batch_size=32)
    return acc


# In[ ]:


def evolve(n, debugging=False):
    if(debugging):
        debug = open("debug.txt", "w")
    else:
        debug = None
    config.load('configReuters')
    # Create 2 separate populations (size is now defined explicitly, but config file can still be used)
    module_pop = population.Population(10, chromosome.ModuleChromo, debug=debug)
    # As the top hierarchical level, the blueprint population needs to be able to see the module population
    blueprint_pop = population.Population(10, chromosome.BlueprintChromo, module_pop, debug=debug)
    # Most of the actual evolving is now handled outside of the population, by CoDeepNEAT
    # Instead of requiring the user to overwrite the evaluation function, CoDeepNEAT evaluates the populations itself,
    # it simply requires a fitness function for the networks it creates passed in as an argument.
    codeepneat.epoch(n, blueprint_pop, module_pop, 10, fitness, data, save_best=True, name='reuters', debug=debug)
    # It will still stop if fitness surpasses the max_fitness_threshold in config file
    # Plots the evolution of the best/average fitness
    visualize.plot_stats(module_pop.stats, name="reutersmod_")
    visualize.plot_stats(blueprint_pop.stats, name="reutersbp_")
    # Visualizes speciation
    #visualize.plot_species(module_pop.species_log, name="NMISTmod_")
    #visualize.plot_species(blueprint_pop.species_log, name="NMISTbp_")


# In[ ]:


evolve(10, True)

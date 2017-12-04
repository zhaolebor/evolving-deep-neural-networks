
# coding: utf-8

# # ICLR18repr demo
# ## CS081 project checkpoint demo
# ### Harsha Uppli, Alan Zhao, Gabriel Meyer-Lee
# 
# The following notebook demonstrates using Hierarchichal Architecture search on MNIST

# In[1]:


from math import pi, floor
import random
import hierarchy
import pickle
import numpy as np
import keras




# ### Flat architectures
# The simplest archecture described in the ICLR paper, technically a hierarchical architecture with layers = 2

# In[3]:


def create_flat_population(pop_size, indiv_size):
    population = []
    for i in range(pop_size):
        population.append(hierarchy.FlatArch(indiv_size))
    return population

def create_hier_population(pop_size, indiv_nodes, indiv_motifs, levels):
    population = []
    for i in range(pop_size):
        population.append(hierarchy.HierArch(indiv_nodes, levels, indiv_motifs))

def assemble_small(architecture, inputdim):
    inputs = keras.layers.Input(shape=inputdim)
    l1 = keras.layers.Conv2D(16, kernel_size=3, activation='relu')(inputs)
    c1 = architecture.assemble(l1, 16)
    l2 = keras.layers.SeparableConv2D(32, kernel_size=3, strides=2, activation='relu')(c1)
    c2 = architecture.assemble(l2, 32)
    l3 = keras.layers.SeparableConv2D(64, kernel_size=3, strides=2, activation='relu')(c2)
    c3 = architecture.assemble(l3, 64)
    l4 = keras.layers.SeparableConv2D(64, kernel_size=3, strides=1, activation='relu')(c3)
    l5 = keras.layers.GlobalAveragePooling2D()(l4)
    outputs = keras.layers.Dense(10, activation='softmax')(l5)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ### Fitness
# 
# For this demonstration we'll be using supervised learning to train the networks produced by the architecture on NMIST and then use their accuracy after 10 epochs as their fitness.

# In[4]:


def fitness(architecture, data):
    network = assemble_small(architecture, data[0].shape[1:])
    network.fit(data[0], data[1],  epochs=5)
    loss, acc = network.evaluate(data[2], data[3])
    architecture.fitness = acc


# ### Random Search
# The authors of this paper report even a simpler search than evolution, simply mutating randomly for a long period and then assessing fitness. This is implemented below.

# In[5]:


def random_mutate(Q, M, steps, data):
    for i in range(steps):
        if len(Q) > 0:
            indiv = Q.pop()
            indiv.mutate()
            fitness(indiv, data)
            M.append(indiv)
        else:
            indiv = random.choice(M).copy()
            indiv.mutate()
            fitness(indiv, data)
            M.append(indiv)




# ### Evolution
# 
# Evolution with CoDeepNEAT is slightly different than evolution with NEAT. The main difference is coevolution, where we have two separate populations with a hierarchical relationship evolving together.

# In[10]:


def tournament_select(pop):
    random.shuffle(pop)
    num_indiv = floor(len(pop)/20)
    contestants = pop[:num_indiv+1]
    contestants.sort()
    contestants.reverse()
    return contestants[0].copy()

def evolve(n, data, debugging=False):
    if(debugging):
        debug = open("debug.txt", "w")
    else:
        debug = None
    Q = create_flat_population(25, 8)
    init_pop = len(Q)
    M = []
    random_mutate(Q,M, init_pop, data)
    for step in range(n-init_pop):
        indiv = tournament_select(M)
        indiv.mutate()
        fitness(indiv, data)
        M.append(indiv)
    




# In[7]:


def eval_best(model_file):
    model = keras.models.load_model(model_file)
    visualize.draw_net(model, "_" + model_file)    
    model.fit(data[0], data[1], epochs=5)
    loss, fitness = model.evaluate(data[2], data[3])
    print("fitness", fitness)


# In[8]:


#eval_best("MNIST_best_model_0")

# ### Sample problem: MNIST data set
# 
# The MNIST data set of handwritten gray scale images of digits 0-9 is a classic computer vision data set and therefore makes for good testing. Conveniently, it's also built into Keras, which our CoDeepNEAT imiplementation is built off of.

def main():
  # In[2]: load data
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  trainx = np.reshape(x_train,(60000,28,28,1))
  testx = np.reshape(x_test, (10000,28,28,1))
  trainy = keras.utils.np_utils.to_categorical(y_train, 10)
  testy = keras.utils.np_utils.to_categorical(y_test, 10)
  data = [trainx, trainy, testx, testy]
  
  # In[7]: random search over flat population
  Q = create_flat_population(25, 8)
  M = []
  random_mutate(Q, M, 100, data)
  M.sort()
  M.reverse()
  for indiv in M:
    print(indiv.fitness)
    
  # In[6]: evolve over flat pop
  #evolve(5, data, False)

if __name__ == "__main__":
  main()


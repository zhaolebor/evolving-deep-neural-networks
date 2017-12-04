
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
import h5py




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
    return population

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
    network.fit(data[0], data[1],  epochs=10)
    loss, acc = network.evaluate(data[2], data[3])
    architecture.fitness = acc
    return acc


# ### Random Search
# The authors of this paper report even a simpler search than evolution, simply mutating randomly for a long period and then assessing fitness. This is implemented below.

# In[5]:


def random_mutate(Q, M, steps, data, record=None):
    for i in range(steps):
        print('\n---------------Mutating step: ' + str(i) +'---------------\n')
        if len(Q) > 0:
            indiv = Q.pop()
            indiv.mutate()
            acc = fitness(indiv, data)
            if record is not None:
                record.write(str(i)+' Fitness: '+str(acc)+'\n')
            M.append(indiv)
        else:
            indiv = random.choice(M).copy()
            indiv.mutate()
            acc = fitness(indiv, data)
            if record is not None:
                record.write(str(i)+' Fitness: '+str(acc)+'\n')
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

def evolve(n, data, record=None):
    Q = create_flat_population(10, 8)
    init_pop = len(Q)
    M = []
    random_mutate(Q,M, init_pop, data, record)
    for step in range(n-init_pop):
        indiv = tournament_select(M)
        indiv.mutate()
        acc = fitness(indiv, data)
        if record is not None:
                record.write(str(step+init_pop)+' Fitness: '+str(acc)+'\n')
        M.append(indiv)




# In[7]:


def eval_best(model_file, record):
    model = keras.models.load_model(model_file)
    visualize.draw_net(model, "_" + model_file)
    model.fit(data[0], data[1], epochs=100)
    loss, fitness = model.evaluate(data[4], data[5])
    record.write("fitness "+str(fitness)+"\n")


def main():
  # In[2]: load data
  (x_train_all, y_train_all), (x_test, y_test) = keras.datasets.cifar10.load_data()
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

  # In[7]: random search over flat population
  record = open('fitnesses.txt','w')
  Q = create_hier_population(10, [4, 5], [6, 1], 3)
  M = []
  random_mutate(Q, M, 100, data, record)
  M.sort()
  M.reverse()
  top = 1
  eval_best(M[0], record)
  record.close()
  for indiv in M[:5]:
    filename = 'model_top_'+str(top)
    indiv.save(filename)
    top += 1


if __name__ == "__main__":
  main()


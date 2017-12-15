
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
import csv
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger




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
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0003, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def assemble_large(architecture, inputdim):
    inputs = keras.layers.Input(shape=inputdim)
    l1 = keras.layers.Conv2D(64, kernel_size=3, activation='relu')(inputs)
    c1 = architecture.assemble(l1, 64)
    l2 = keras.layers.SeparableConv2D(64, kernel_size=3, strides=1, activation='relu')(c1)
    c2 = architecture.assemble(l2, 64)
    l3 = keras.layers.SeparableConv2D(128, kernel_size=3, strides=2, activation='relu')(c2)
    c3 = architecture.assemble(l3, 128)
    l4 = keras.layers.SeparableConv2D(128, kernel_size=3, strides=1, activation='relu')(c3)
    c4 = architecture.assemble(l4, 128)
    l5 = keras.layers.SeparableConv2D(256, kernel_size=3, strides=2, activation='relu')(c2)
    c5 = architecture.assemble(l5, 256)
    l6 = keras.layers.SeparableConv2D(256, kernel_size=3, strides=1, activation='relu')(c2)
    c6 = architecture.assemble(l6, 256)
    l7 = keras.layers.SeparableConv2D(256, kernel_size=3, strides=1, activation='relu')(c2)
    l8 = keras.layers.GlobalAveragePooling2D()(l7)
    outputs = keras.layers.Dense(10, activation='softmax')(l8)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0003, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# ### Fitness
#
# For this demonstration we'll be using supervised learning to train the networks produced by the architecture on NMIST and then use their accuracy after 10 epochs as their fitness.

# In[4]:


def fitness(architecture, data):

    #csv_logger = CSVLogger('random_sample.csv', append=True)
    datafile = 'random_sample.csv'
    crop_size = 24
    batch_size = 256
    num_epoch = 200
    eval_network = assemble_large(architecture, data[0].shape[1:])
    train_network = assemble_large(architecture.copy(), (crop_size, crop_size, data[0].shape[-1]))


    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    #network.fit_generator(datagen.flow(data[0], data[1], batch_size=batch_size),
    #                    steps_per_epoch=data[0].shape[0] // batch_size,
    #                    validation_data=(data[2], data[3]),
    #                    epochs=num_epoch, verbose=1, max_queue_size=100)

    # assemble a new training set
    new_x_train = []
    for x in data[0]:
        cropped_x = random_crop(x, crop_size)
        new_x_train.append(cropped_x)
    new_x_train = np.array(new_x_train)

    new_x_test = []
    for x in data[2]:
        cropped_x = random_crop(x, crop_size)
        new_x_test.append(cropped_x)
    new_x_test = np.array(new_x_test)


    for e in range(num_epoch):
        print('Epoch:', e)
        batches = 0
        for x_batch, y_batch in datagen.flow(data[0], data[1], batch_size=batch_size):
            new_x_batch = []
            for x in x_batch:
                cropped_x = random_crop(x, crop_size)
                new_x_batch.append(cropped_x)
            new_x_batch = np.array(new_x_batch)
            train_network.fit(new_x_batch, y_batch, batch_size=batch_size, verbose=0)
            batches += 1
            if batches >= len(data[0]) / batch_size:
            # we need to break the loop by hand because
            # the generator loops indefinitely
                break

        # evaluate the network on a training set
        train_loss, train_acc = train_network.evaluate(new_x_train[:10000], data[1][:10000], batch_size=batch_size, verbose=0)
        test_loss, test_acc = train_network.evaluate(new_x_test, data[3], batch_size=batch_size, verbose=0)

        with open(datafile, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            row = [e, train_loss, train_acc, test_loss, test_acc]
            writer.writerow(row)

    weights = train_network.get_weights()
    eval_network.set_weights(weights)

    loss, acc = eval_network.evaluate(data[2], data[3])
    with open(datafile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        row = [loss, acc, eval_network.count_params()]
        writer.writerow(row)
    architecture.fitness = acc
    return acc, eval_network.count_params()


def random_crop(image, crop_size):
    height, width = image.shape[0:2]
    dx = crop_size
    dy = crop_size
    if width < dx or height < dy:
        return None
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return image[y:(y+dy), x:(x+dx),:]


# ### Random Search
# The authors of this paper report even a simpler search than evolution, simply mutating randomly for a long period and then assessing fitness. This is implemented below.

# In[5]:


def random_mutate(Q, M, steps, data, record=None):
    for i in range(steps):
        print('\n---------------Mutating step: ' + str(i) +'---------------\n')
        if len(Q) > 0:
            indiv = Q.pop()
            acc, num_params = fitness(indiv, data)
            if record is not None:
                f = open(record,'a')
                f.write(str(i)+' Fitness: '+str(acc)+' Params: '+str(num_params)+'\n')
                f.close()
            M.append(indiv)
        else:
            indiv = random.choice(M).copy()
            indiv.mutate()
            acc, num_params = fitness(indiv, data)
            if record is not None:
                f = open(record,'a')
                f.write(str(i)+' Fitness: '+str(acc)+' Params: '+str(num_params)+'\n')
                f.close()
            M.append(indiv)

def initial_mutate(pop):
    large_number = 100
    for indiv in pop:
        for i in range(large_number):
            indiv.mutate()


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

def evolve(Q, n, data, record=None):
    init_pop = len(Q)
    M = []
    random_mutate(Q,M, init_pop, data, record)
    print(len(M))
    M.sort()
    M.reverse()
    best = M[0]
    file = open('best_random_arch','wb')
    pickle.dump(best, file)
    file.close()
    for step in range(init_pop, n):
        indiv = tournament_select(M)
        indiv.mutate()
        acc, num_params = fitness(indiv, data)
        if acc > best.fitness:
            best = indiv
            file = open('best_evol_arch','wb')
            pickle.dump(best, file)
            file.close()
        if record is not None:
                f = open(record,'a')
                f.write(str(step)+' Fitness: '+str(acc)+' Params: '+str(num_params)+'\n')
                f.close()
        M.append(indiv)



#eval_best("MNIST_best_model_0")

# ### Sample problem: MNIST data set
#
# The MNIST data set of handwritten gray scale images of digits 0-9 is a classic computer vision data set and therefore makes for good testing. Conveniently, it's also built into Keras, which our CoDeepNEAT imiplementation is built off of.

def main():
    # In[2]: load data
    (x_train_all, y_train_all), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train_all = y_train_all[:,0]
    y_test = y_test[:,0]

    x_train_all = np.reshape(x_train_all, (x_train_all.shape[0], 32, 32, 3)).astype('float32') / 255
    x_test = np.reshape(x_test, (x_test.shape[0], 32, 32, 3)).astype('float32') / 255
    y_train_all_one_hot = keras.utils.np_utils.to_categorical(y_train_all)
    y_test = keras.utils.np_utils.to_categorical(y_test)


    data = [x_train_all, y_train_all_one_hot, x_test, y_test]
    print("data shapes")
    print("  x train:", x_train_all.shape)
    print("  y train:", y_train_all_one_hot.shape)

    print("  x test:", x_test.shape)
    print("  y test:", y_test.shape)

    # In[7]: random search over flat population
    record = 'single.txt'
    Q = create_hier_population(1, [4, 5], [6, 1], 3)
    initial_mutate(Q)
    arch = Q[0]
    file = open('single_arch','wb')
    pickle.dump(arch, file)
    file.close()
    acc, param = fitness(arch, data)
    print(acc, param)

    # In[6]: evolve over flat pop
    #evolve(5, data, False)

if __name__ == "__main__":
    main()

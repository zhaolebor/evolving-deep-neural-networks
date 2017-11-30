"""
Implements graph based hierarchical convolutional network encoding, as found in the anonymous
paper submitted to ICLR 2018
"""

import keras
import random
import numpy as np

class Motif(object):
    def __init__(self, num_nodes, prev_motifs=[], level=2):
        if level < 0:
            raise ValueError('Invalid level value')
        self.__size = num_nodes
        self.__level = level
        # initialize node data as nothing
        self.__x = []
        # if lowest non primitive level operations are primitive
        if level == 2:
            self.__o = [ None,'no_op', 'ident','1x1','3x3_depth','3x3_sep', 'max_pool', 'avg_pool']
        # otherwise build operations from previous level of motifs
        else:
            self.__o = [ None, 'no_op', 'ident' ]
            self.__o.extend(prev_motifs)
        # initialize adjacency matrix to encode linear chain of identity connections
        ones = np.ones((num_nodes, num_nodes), dtype=int)
        self.__G = np.tril(ones, -1)
        for j in range(1,num_nodes):
            self.__G.itemset((j,j-1), 2)
    level = property(lambda self: self.__level)

    def decode(self, inputs, c):
        self.__x.append(inputs)
        if self.__level > 2:
            for i in range(1, self.__size):
                new_x = []
                for j in range(i):
                    op = self.__o[self.__G.item((i,j))]
                    if op is None:
                        raise IndexError('Attempt to access invalid graph edge')
                    elif op == 'no_op':
                        continue
                    elif op == 'ident':
                        new_x.append(self.__x[j])
                    else:
                        new_x.append(op.decode(self.__x[j]))
                if len(new_x) == 0:
                    self.__G.itemset((i,i-1), 2)
                    self.__x.append(self.__x[i-1])
                elif len(new_x) == 1:
                    self.__x.append(new_x[0])
                else:
                    self.__x.append(keras.layers.Concatenate()(new_x))
        elif self.__level == 2:
            for i in range(1, self.__size):
                new_x = []
                for j in range(i):
                    op = self.__o[self.__G.item((i,j))]
                    if op is None or op == 'no_op':
                        continue
                    elif op == 'ident':
                        new_x.append(self.__x[j])
                    elif op == '1x1':
                        x_temp = keras.layers.Conv2D(c, 1, padding='same', activation='relu')(self.__x[j])
                        new_x.append(keras.layers.BatchNormalization()(x_temp))
                    elif op == '3x3_depth':
                        x_temp = keras.layers.Conv2D(c, 3, padding='same', activation='relu')(self.__x[j])
                        new_x.append(keras.layers.BatchNormalization()(x_temp))
                    elif op == '3x3_sep':
                        x_temp = keras.layers.SeparableConv2D(c, 3, padding='same', activation='relu')(self.__x[j])
                        new_x.append(keras.layers.BatchNormalization()(x_temp))
                    elif op == 'max_pool':
                        new_x.append(keras.layers.MaxPooling2D(3, padding='same')(self.__x[j]))
                    elif op == 'avg_pool':
                        new_x.append(keras.layers.AveragePooling2D(3, padding='same')(self.__x[j]))
                    else:
                        raise ValueError('Invalid level 1 motif: '+op)
                if len(new_x) == 0:
                    self.__G.itemset((i,i-1), 2)
                    self.__x.append(self.__x[i-1])
                elif len(new_x) == 1:
                    self.__x.append(new_x[0])
                else:
                    self.__x.append(keras.layers.Concatenate()(new_x))
        return self.__x[-1]

    def mutate(self):
        i = random.randrange(1, self.__size)
        j = random.randrange(i)
        self.__G.itemset((i,j), random.randrange(1,len(self.__o)))

class Architecture(object):
    def __init__(self, num_nodes):
        self._num_nodes = num_nodes
        self.fitness = 0

    def __cmp__(self, other):
        return cmp(self.fitness, other.fitness)

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def mutate(self):
        raise NotImplementedError

    def assemble(self, inputs):
        raise NotImplementedError

class FlatArch(Architecture):
    def __init__(self, num_nodes):
        super(FlatArch, self).__init__(num_nodes)
        self.__m = Motif(num_nodes)

    def mutate(self):
        self.__m.mutate()

    def assemble(self, inputs, c):
        output = self.__m.decode(inputs, c)
        return output


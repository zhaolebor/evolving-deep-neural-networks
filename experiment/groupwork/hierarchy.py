"""
Implements graph based hierarchical convolutional network encoding, as found in the anonymous
paper submitted to ICLR 2018
"""

import keras
import random
from chromosome import Chromosome
import numpy as np

class Motif(Chromosome):
    def __init__(self, parent1_id, parent2_id, num_nodes, level=2, gene_type='MOTIF'):
        Chromosome.__init__(parent1_id, parent2_id, gene_type)
        if level < 2:
            raise ValueError('Invalid level value')
        self._size = num_nodes
        self._level = level
        self._G = None

    level = property(lambda self: self._level)
    genes = property(lambda self: self._G)

    def decode(self, inputs):
        raise NotImplementedError

    def mutate(self):
        raise NotImplementedError

    def set_genes(self):
        self._genes = self._G.nonzero()

class ConvMotif(Motif):
    def __init__(self, parent1_id, parent2_id, num_nodes, prev_motifs=[], level=2):
        super(ConvMotif, self).__init__(parent1_id, parent2_id, num_nodes, level)
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
        self._G = np.tril(ones, -1)
        for j in range(1,num_nodes):
            self._G.itemset((j,j-1), 2)
        set_genes()

    def decode(self, inputs, c):
        self.__x = []
        self.__x.append(inputs)
        if self._level > 2:
            for i in range(1, self._size):
                new_x = []
                for j in range(i):
                    op = self.__o[self._G.item((i,j))]
                    if op is None:
                        raise IndexError('Attempt to access invalid graph edge')
                    elif op == 'no_op':
                        continue
                    elif op == 'ident':
                        new_x.append(self.__x[j])
                    else:
                        new_x.append(op.decode(self.__x[j], c))
                if len(new_x) == 0:
                    self._G.itemset((i,i-1), 2)
                    self.__x.append(self.__x[i-1])
                elif len(new_x) == 1:
                    self.__x.append(new_x[0])
                else:
                    self.__x.append(keras.layers.Concatenate()(new_x))
        elif self._level == 2:
            for i in range(1, self._size):
                new_x = []
                for j in range(i):
                    op = self.__o[self._G.item((i,j))]
                    if op is None or op == 'no_op':
                        continue
                    elif op == 'ident':
                        new_x.append(self.__x[j])
                    elif op == '1x1':
                        x_temp = keras.layers.Conv2D(c, 1, padding='same', activation='relu')(self.__x[j])
                        new_x.append(keras.layers.BatchNormalization()(x_temp))
                    elif op == '3x3_depth':
                        x_temp = keras.layers.Conv2D(c, 3, strides=1 padding='same', activation='relu')(self.__x[j])
                        new_x.append(keras.layers.BatchNormalization()(x_temp))
                    elif op == '3x3_sep':
                        x_temp = keras.layers.SeparableConv2D(c, 3, strides=1 padding='same', activation='relu')(self.__x[j])
                        new_x.append(keras.layers.BatchNormalization()(x_temp))
                    elif op == 'max_pool':
                        new_x.append(keras.layers.MaxPooling2D(3, strides=1, padding='same')(self.__x[j]))
                    elif op == 'avg_pool':
                        new_x.append(keras.layers.AveragePooling2D(3, strides=1, padding='same')(self.__x[j]))
                    else:
                        raise ValueError('Invalid level 1 motif: '+op)
                if len(new_x) == 0:
                    self._G.itemset((i,i-1), 2)
                    self.__x.append(self.__x[i-1])
                elif len(new_x) == 1:
                    self.__x.append(new_x[0])
                else:
                    self.__x.append(keras.layers.Concatenate()(new_x))
        assert len(self.__x) == self._size, 'Size Error: ' + str(len(self.__x)) + ' ' + str(self._size)
        return self.__x[-1]

    def mutate(self):
        i = random.randrange(1, self._size)
        j = random.randrange(i)
        self._G.itemset((i,j), random.randrange(1,len(self.__o)))
        set_genes()

class DenseMotif(Motif):
    def __init__(self, num_nodes, prev_motifs=[], level=2):
        super(ConvMotif, self).__init__(num_nodes, level)
        # initialize node data as nothing
        self.__x = []
        # if lowest non primitive level operations are primitive
        if level == 2:
            self.__o = [ None,'no_op', 'ident', 'dense']
        # otherwise build operations from previous level of motifs
        else:
            self.__o = [ None, 'no_op', 'ident' ]
            self.__o.extend(prev_motifs)
        # initialize adjacency matrix to encode linear chain of identity connections
        ones = np.ones((num_nodes, num_nodes), dtype=int)
        self._G = np.tril(ones, -1)
        for j in range(1,num_nodes):
            self._G.itemset((j,j-1), 2)

    def decode(self, inputs, c):
        self.__x = []
        self.__x.append(inputs)
        if self._level > 2:
            for i in range(1, self._size):
                new_x = []
                for j in range(i):
                    op = self.__o[self._G.item((i,j))]
                    if op is None:
                        raise IndexError('Attempt to access invalid graph edge')
                    elif op == 'no_op':
                        continue
                    elif op == 'ident':
                        new_x.append(self.__x[j])
                    else:
                        new_x.append(op.decode(self.__x[j], c))
                if len(new_x) == 0:
                    self._G.itemset((i,i-1), 2)
                    self.__x.append(self.__x[i-1])
                elif len(new_x) == 1:
                    self.__x.append(new_x[0])
                else:
                    self.__x.append(keras.layers.Concatenate()(new_x))
        elif self._level == 2:
            for i in range(1, self._size):
                new_x = []
                for j in range(i):
                    op = self.__o[self._G.item((i,j))]
                    if op is None or op == 'no_op':
                        continue
                    elif op == 'ident':
                        new_x.append(self.__x[j])
                    elif op == 'dense':
                        x_temp = keras.layers.Dense(c, activation='relu')(self.__x[j])
                        new_x.append(keras.layers.BatchNormalization()(x_temp))
                    else:
                        raise ValueError('Invalid level 1 motif: '+op)
                if len(new_x) == 0:
                    self._G.itemset((i,i-1), 2)
                    self.__x.append(self.__x[i-1])
                elif len(new_x) == 1:
                    self.__x.append(new_x[0])
                else:
                    self.__x.append(keras.layers.Concatenate()(new_x))
        assert len(self.__x) == self._size, 'Size Error: ' + str(len(self.__x)) + ' ' + str(self._size)
        return self.__x[-1]

    def mutate(self):
        i = random.randrange(1, self._size)
        j = random.randrange(i)
        self._G.itemset((i,j), random.randrange(1,len(self.__o)))
        set_genes()


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
    def __init__(self, num_nodes, input_dim, motifs=None):
        super(FlatArch, self).__init__(num_nodes)
        if motifs is not None:
            self.__m = motifs
        else:
            if input_dim == 3:
                self.__m = ConvMotif(num_nodes)
            else:
                self.__m = DenseMotif(num_nodes)

    def mutate(self):
        self.__m.mutate()

    def assemble(self, inputs, c):
        output = self.__m.decode(inputs, c)
        return output

    def copy(self):
        return FlatArch(self._num_nodes, motifs=self.__m)

class HierArch(Architecture):
    def __init__(self, num_nodes, num_levels, num_motifs, input_dim, motifs=None):
        super(HierArch, self).__init__(num_nodes)
        assert num_levels > 2, "Insufficent levels to require hierarchical architecture"
        self.height = num_levels
        self._num_motifs = num_motifs
        self.__m = []
        if motifs is not None:
            self.__m = motifs
        else:
            for i in range(0, num_levels-1):
                self.__m.insert(i,[])
                for j in range(num_motifs[i]):
                    print(num_nodes[i])
                    if i == 0:
                        if input_dim == 3:
                            self.__m = ConvMotif(num_nodes[i])
                        else:
                            self.__m = DenseMotif(num_nodes[i])
                    else:
                        if input_dim == 3:
                            self.__m[i].append(ConvMotif(num_nodes[i], prev_motifs = self.__m[i-1], level=i+2))
                        else:
                            self.__m[i].append(DenseMotif(num_nodes[i], prev_motifs = self.__m[i-1], level=i+2))

    def mutate(self):
        l = random.randint(2, self.height)
        m_ind = random.randrange(self._num_motifs[l-2])
        m = self.__m[l-2][m_ind]
        m.mutate()

    def assemble(self, inputs, c):
        output = self.__m[self.height-2][0].decode(inputs, c)
        return output

    def copy(self):
        return HierArch(self._num_nodes, self.height, self._num_motifs, motifs=self.__m)



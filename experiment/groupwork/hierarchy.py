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
        self._genes = None
        self._num_params = 0

    level = property(lambda self: self._level)
    genes = property(lambda self: self._genes)

    def size(self):
        return self._num_params

    def distance(self, other):
        diff_array = np.equal(self._genes, other.genes)
        diff = diff_array.sum()
        return diff
    
    @class method
    def create_initial(cls, num_nodes, lower_pop=None):
        if lower_pop is not None:
          c = cls(None, None, num_nodes, lower_pop, level=3)
        else:
          c = cls(None, None, num_nodes)
        
class ConvoMotif(Motif):
    def __init__(self, parent1_id, parent2_id, num_nodes, prev_motifs=[], level=2):
        super(ConvoMotif, self).__init__(parent1_id, parent2_id, num_nodes, level)
        # if lowest non primitive level operations are primitive
        if level == 2:
            self.__o = [ None,'no_op', 'ident','1x1','3x3_depth','3x3_sep', 'max_pool', 'avg_pool']
        # otherwise build operations from previous level of motifs
        else:
            self.__o = [ None, 'no_op', 'ident' ]
            self.__o.extend(prev_motifs)
        # initialize adjacency matrix to encode linear chain of identity connections
        ones = np.ones((num_nodes, num_nodes), dtype=int)
        self._genes = np.tril(ones, -1)
        for j in range(1,num_nodes):
            self._genes.itemset((j,j-1), 2)

    def decode(self, inputs, c):
        x_tens = []
        x_tens.append(inputs)
        if self._level > 2:
            for i in range(1, self._size):
                new_x = []
                for j in range(i):
                    op = self.__o[self._genes.item((i,j))]
                    if op is None:
                        raise IndexError('Attempt to access invalid graph edge')
                    elif op == 'no_op':
                        continue
                    elif op == 'ident':
                        new_x.append(x_tens[j])
                    else:
                        new_x.append(op.decode(x_tens[j], c))
                if len(new_x) == 0:
                    self._genes.itemset((i,i-1), 2)
                    x_tens.append(x_tens[i-1])
                elif len(new_x) == 1:
                    x_tens.append(new_x[0])
                else:
                    x_tens.append(keras.layers.Concatenate()(new_x))
        elif self._level == 2:
            for i in range(1, self._size):
                new_x = []
                for j in range(i):
                    op = self.__o[self._genes.item((i,j))]
                    if op is None or op == 'no_op':
                        continue
                    elif op == 'ident':
                        new_x.append(x_tens[j])
                    elif op == '1x1':
                        x_temp = keras.layers.Conv2D(c, 1, padding='same', activation='relu')(x_tens[j])
                        new_x.append(keras.layers.BatchNormalization()(x_temp))
                    elif op == '3x3_depth':
                        x_temp = keras.layers.Conv2D(c, 3, strides=1 padding='same', activation='relu')(x_tens[j])
                        new_x.append(keras.layers.BatchNormalization()(x_temp))
                    elif op == '3x3_sep':
                        x_temp = keras.layers.SeparableConv2D(c, 3, strides=1 padding='same', activation='relu')(x_tens[j])
                        new_x.append(keras.layers.BatchNormalization()(x_temp))
                    elif op == 'max_pool':
                        new_x.append(keras.layers.MaxPooling2D(3, strides=1, padding='same')(x_tens[j]))
                    elif op == 'avg_pool':
                        new_x.append(keras.layers.AveragePooling2D(3, strides=1, padding='same')(x_tens[j]))
                    else:
                        raise ValueError('Invalid level 1 motif: '+op)
                if len(new_x) == 0:
                    self._genes.itemset((i,i-1), 2)
                    x_tens.append(x_tens[i-1])
                elif len(new_x) == 1:
                    x_tens.append(new_x[0])
                else:
                    x_tens.append(keras.layers.Concatenate()(new_x))
        assert len(x_tens) == self._size, 'Size Error: ' + str(len(x_tens)) + ' ' + str(self._size)
        return x_tens[-1]

    def mutate(self):
        i = random.randrange(1, self._size)
        j = random.randrange(i)
        self._genes.itemset((i,j), random.randrange(1,len(self.__o)))

class RecurMotif(Motif):
    def __init__(self, parent1_id, parent2_id, num_nodes, prev_motifs=[], level=2):
        super(RecurMotif, self).__init__(parent1_id, parent2_id, num_nodes, level)
        # if lowest non primitive level operations are primitive
        if level == 2:
            self.__o = [ None,'no_op', 'ident','rnn','gru','lstm', 'conv1d', 'avg_pool', 'max_pool']
        # otherwise build operations from previous level of motifs
        else:
            self.__o = [ None, 'no_op', 'ident' ]
            self.__o.extend(prev_motifs)
        # initialize adjacency matrix to encode linear chain of identity connections
        ones = np.ones((num_nodes, num_nodes), dtype=int)
        self._genes = np.tril(ones, -1)
        for j in range(1,num_nodes):
            self._genes.itemset((j,j-1), 2)

    def decode(self, inputs, c):
        x_tens = []
        x_tens.append(inputs)
        if self._level > 2:
            for i in range(1, self._size):
                new_x = []
                for j in range(i):
                    op = self.__o[self._genes.item((i,j))]
                    if op is None:
                        raise IndexError('Attempt to access invalid graph edge')
                    elif op == 'no_op':
                        continue
                    elif op == 'ident':
                        new_x.append(x_tens[j])
                    else:
                        new_x.append(op.decode(x_tens[j], c))
                if len(new_x) == 0:
                    self._genes.itemset((i,i-1), 2)
                    x_tens.append(x_tens[i-1])
                elif len(new_x) == 1:
                    x_tens.append(new_x[0])
                else:
                    x_tens.append(keras.layers.Concatenate()(new_x))
        elif self._level == 2:
            for i in range(1, self._size):
                new_x = []
                for j in range(i):
                    op = self.__o[self._genes.item((i,j))]
                    if op is None or op == 'no_op':
                        continue
                    elif op == 'ident':
                        new_x.append(x_tens[j])
                    elif op == 'rnn':
                        new_x.append(keras.layers.SimpleRNN(c, return_sequences=True, stateful=True)(x_tens[j]))
                    elif op == 'gru':
                        new_x.append(keras.layers.GRU(c, return_sequences=True, stateful=True)(x_tens[j]))
                    elif op == 'lstm':
                        new_x.append(keras.layers.LSTM(c, return_sequences=True, stateful=True)(x_tens[j]))
                    elif op == 'max_pool':
                        new_x.append(keras.layers.MaxPooling1D(3, strides=1, padding='same')(x_tens[j]))
                    elif op == 'conv1d':
                        new_x.append(keras.layers.Conv1D(c, 3, strides=1, padding='same')(x_tens[j]))
                    elif op == 'avg_pool':
                        new_x.append(keras.layers.AveragePooling2D(3, strides=1, padding='same')(x_tens[j]))
                    else:
                        raise ValueError('Invalid level 1 motif: '+op)
                if len(new_x) == 0:
                    self._genes.itemset((i,i-1), 2)
                    x_tens.append(x_tens[i-1])
                elif len(new_x) == 1:
                    x_tens.append(new_x[0])
                else:
                    x_tens.append(keras.layers.Concatenate()(new_x))
        assert len(x_tens) == self._size, 'Size Error: ' + str(len(x_tens)) + ' ' + str(self._size)
        return x_tens[-1]

    def mutate(self):
        i = random.randrange(1, self._size)
        j = random.randrange(i)
        self._genes.itemset((i,j), random.randrange(1,len(self.__o)))

class DenseMotif(Motif):
    def __init__(self, num_nodes, prev_motifs=[], level=2):
        super(ConvMotif, self).__init__(num_nodes, level)
        # if lowest non primitive level operations are primitive
        if level == 2:
            self.__o = [ None,'no_op', 'ident', 'dense']
        # otherwise build operations from previous level of motifs
        else:
            self.__o = [ None, 'no_op', 'ident' ]
            self.__o.extend(prev_motifs)
        # initialize adjacency matrix to encode linear chain of identity connections
        ones = np.ones((num_nodes, num_nodes), dtype=int)
        self._genes = np.tril(ones, -1)
        for j in range(1,num_nodes):
            self._genes.itemset((j,j-1), 2)

    def decode(self, inputs, c):
        x_tens = []
        x_tens.append(inputs)
        if self._level > 2:
            for i in range(1, self._size):
                new_x = []
                for j in range(i):
                    op = self.__o[self._genes.item((i,j))]
                    if op is None:
                        raise IndexError('Attempt to access invalid graph edge')
                    elif op == 'no_op':
                        continue
                    elif op == 'ident':
                        new_x.append(x_tens[j])
                    else:
                        new_x.append(op.decode(x_tens[j], c))
                if len(new_x) == 0:
                    self._genes.itemset((i,i-1), 2)
                    x_tens.append(x_tens[i-1])
                elif len(new_x) == 1:
                    x_tens.append(new_x[0])
                else:
                    x_tens.append(keras.layers.Concatenate()(new_x))
        elif self._level == 2:
            for i in range(1, self._size):
                new_x = []
                for j in range(i):
                    op = self.__o[self._genes.item((i,j))]
                    if op is None or op == 'no_op':
                        continue
                    elif op == 'ident':
                        new_x.append(x_tens[j])
                    elif op == 'dense':
                        x_temp = keras.layers.Dense(c, activation='relu')(x_tens[j])
                        new_x.append(keras.layers.BatchNormalization()(x_temp))
                    else:
                        raise ValueError('Invalid level 1 motif: '+op)
                if len(new_x) == 0:
                    self._genes.itemset((i,i-1), 2)
                    x_tens.append(x_tens[i-1])
                elif len(new_x) == 1:
                    x_tens.append(new_x[0])
                else:
                    x_tens.append(keras.layers.Concatenate()(new_x))
        assert len(x_tens) == self._size, 'Size Error: ' + str(len(x_tens)) + ' ' + str(self._size)
        return x_tens[-1]

    def mutate(self):
        i = random.randrange(1, self._size)
        j = random.randrange(i)
        self._genes.itemset((i,j), random.randrange(1,len(self.__o)))


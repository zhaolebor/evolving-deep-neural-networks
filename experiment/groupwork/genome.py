# -*- coding: UTF-8 -*-
import random
from .config import Config
import keras
import math

class ConvMotif(object):
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
                        new_x.append(op.decode(self.__x[j], c))
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
                        new_x.append(keras.layers.MaxPooling2D(3, strides=1, padding='same')(self.__x[j]))
                    elif op == 'avg_pool':
                        new_x.append(keras.layers.AveragePooling2D(3, strides=1, padding='same')(self.__x[j]))
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


class LayerGene(object):
    _id = 0
    def __init__(self, id, layertype, outputdim):
        """
        A layer gene is a node which represents a single Keras layer.
        """
        if (id == None):
            self._id = self.__get_new_id()
        else:
            self._id = id
        self._type = layertype
        self._size = outputdim

    id = property(lambda self: self._id)
    type = property(lambda self: self._type)
    size = property(lambda self: self._size)

    @classmethod
    def __get_new_id(cls):
        cls._id += 1
        return cls._id


    def __str__(self):
        return "Layer ID: %2d Type: %6s Size: %4d" \
                %(self._id, self._type, self._size)

    def get_child(self, other):
        """
        Creates a new LayerGene randonly inheriting its attributes from
        parents.
        """
        if(self._type != other._type):
            raise TypeError

        g = LayerGene(self.__get_new_id, self._type, random.choice(self._size, other._size))
        return g

    def __cmp__(self, other):
        return cmp(self._id, other._id)

    def __lt__(self, other):
        return self._id <  other._id

    def __gt__(self, other):
        return self._id > other._id


    def copy(self):
        return LayerGene(self._id, self._type, self._size)

    def mutate(self):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

class DenseGene(LayerGene):

    layer_params = {
            "_size": [2**i for i in range(4, int(math.log(256, 2)) + 1)],
            "_activation": ['sigmoid', 'tanh', 'relu'],
            "_dropout": [0.1*i for i in range(8)],
            "_batch_norm": [True, False],
    }
    def __init__(self, id, numnodes, activation='relu', dropout=0.0, batch_norm=True, layertype='DENSE'):
        super(DenseGene, self).__init__(id, layertype, numnodes)
        self._activation = activation
        self._dropout = dropout
        self._batch_norm = batch_norm

    def get_child(self, other):
        if(self._type != other._type):
            raise TypeError
        child_param = []
        for key in list(DenseGene.layer_params.keys()):
           child_param.append(random.choice(getattr(self,key), getattr(other,key)))
        return DenseGene(self.__get_new_id(), child_param[0], child_param[1], child_param[2], \
                child_param[3])

    def copy(self):
        return DenseGene(self._id, self._size, self._activation, \
                self._dropout, self._batch_norm)

    def mutate(self):
        r = random.random
        if r() < .5:
            self._dropout = random.choice(self.layer_params['_dropout'])
        else:
            self._size = random.choice(self.layer_params['_size'])

    def decode(self, x):
        inputdim = len(keras.backend.int_shape(x)[1:])
        if inputdim > 2:
            x = keras.layers.Flatten()(x)
        if self._dropout:
            x = keras.layers.Dropout(self._dropout)(x)
        x = keras.layers.Dense(self._size, activation=self._activation)(x)
        if self._batch_norm:
            x = keras.layers.BatchNormalization()(x)
        return x

class ConvGene(LayerGene):

    layer_params = {
            "_size": [2**i for i in range(4, 9)],
            "_kernel_size": [1,3,5],
            "_activation": ['sigmoid','tanh','relu'],
            "_dropout": [.1*i for i in range(8)],
            "_padding": ['same','valid'],
            "_strides": [(1,1), (2,1), (1,2), (2,2)],
            "_max_pooling": [True, False],
            "_batch_norm": [True, False],
    }

    def __init__(self, id, numfilter, kernel_size=1, activation='relu', dropout=0.0, \
            padding='same', strides=(1,1), max_pooling=False, batch_norm=True, layertype='CONV'):
        super(ConvGene, self).__init__(id, layertype, numfilter)
        self._kernel_size = kernel_size
        self._activation = activation
        self._dropout = dropout
        self._padding = padding
        self._strides = strides
        self._max_pooling = max_pooling
        self._batch_norm = batch_norm

    def get_child(self, other):
        if(self._type != other._type):
            raise TypeError
        child_param = []
        for key in list(ConvGene.layer_params.keys()):
           child_param.append(random.choice(getattr(self,key), getattr(other,key)))
        return ConvGene(self.__get_new_id(), child_param[0], child_param[1], child_param[2], \
                child_param[3], child_param[4], child_param[5], child_param[6], child_param[7])

    def copy(self):
        return ConvGene(self._id, self._size, self._kernel_size, self._activation, self._dropout,self._padding, self._strides, self._max_pooling, self._batch_norm)

    def mutate(self):
        r = random.random()
        if r < .25:
            self._max_pooling = random.choice(self.layer_params['_max_pooling'])
        elif r < .5:
            self._dropout = random.choice(self.layer_params['_dropout'])
        elif r < .75:
            self._kernel_size = random.choice(self.layer_params['_kernel_size'])
        else:
            self._size = random.choice(self.layer_params['_size'])

    def decode(self, x):
        # the below code attempts to reconstruct a 2D shape from a 1D input. This is probably
        # a bad idea and will be removed soon
        inputdim = len(keras.backend.int_shape(x)[1:])
        if inputdim == 1:
            xval = keras.backend.int_shape(x)[1]
            dim1 = math.floor(math.sqrt(xval))
            while (xval%dim1 != 0):
                dim1 -= 1
            dim2 = xval/dim1
            x = keras.layers.Reshape((int(dim1),int(dim2),1))(x)
        if self._dropout:
            x = keras.layers.Dropout(self._dropout)(x)
        x = keras.layers.Conv2D(self._size, self._kernel_size, strides=self._strides, \
                padding=self._padding, activation=self._activation)(x)
        if self._max_pooling:
            if keras.backend.int_shape(x)[1] > 1:
              x = keras.layers.MaxPool2D()(x)
        return x

class LSTMGene(LayerGene):

    layer_params = {
            "_size": [2**i for i in range(6, 10)],
            "_activation": ['sigmoid', 'tanh', 'relu'],
            "_dropout": [0.1*i for i in range(7)],
    }
    def __init__(self, id, numnodes, activation='tanh', dropout=0.0, layertype='LSTM'):
        super(LSTMGene, self).__init__(id, layertype, numnodes)
        self._activation = activation
        self._dropout = dropout

    def get_child(self, other):
        if(self._type != other._type):
            raise TypeError
        child_param = []
        for key in list(LSTMGene.layer_params.keys()):
           child_param.append(random.choice(getattr(self,key), getattr(other,key)))
        return LSTMGene(self.__get_new_id(), child_param[0], child_param[1], child_param[2])

    def copy(self):
        return LSTMGene(self._id, self._size, self._activation, \
                self._dropout)

    def mutate(self):
        r = random.random()
        if r < .5:
            self._dropout = random.choice(self.layer_params['_dropout'])
        else:
            self._size = random.choice(self.layer_params['_size'])

    def decode(self, x):
        # TODO remove hardcoded batch size
        inputdim = len(keras.backend.int_shape(x)[1:])
        if self._dropout:
            x = keras.layers.Dropout(self._dropout)(x)
        if inputdim > 3:
            x = keras.layers.ConvLSTM2D(self._size, activation=self._activation, return_sequences=True, batch_size=32)(x)
        else:
            x = keras.layers.LSTM(self._size, activation=self._activation, return_sequences=True, batch_size=32)(x)
        return x


class ModuleGene(object):
    _id = 0
    def __init__(self, id, modspecies, modtype='DENSE'):
        """
        A module gene is a node which represents a multilayer component of
        a deep neural network.
        """
        if (id == None):
            self._id = self.__get_new_id()
        else:
            self._id = id
        self._type = modtype
        self._module = modspecies

    @classmethod
    def __get_new_id(cls):
        cls._id += 1
        return cls._id

    id = property(lambda self: self._id)
    type = property(lambda self: self._type)
    module = property(lambda self: self._module)

    def __str__(self):
        return "Gene ID: %2d Species ID: %2d"%(self._id, self._module.id)

    def get_child(self, other):
        """
        Creates a new NodeGene randomly inheriting its attributes from
        parents.
        """
        if (self._type != other._type):
            raise TypeError

        g = ModuleGene(self.__get_new_id(), random.choice(self._module, other._module))
        return g

    def copy(self):
        return ModuleGene(self._id, self._module)

    def set_module(self, modspecies):
        if modspecies.members[0].type != self._type:
            raise TypeError
        self._module = modspecies

    def mutate(self):
        pass


class Connection(object):
    def __init__(self, innodes=[], outnodes=[]):
        self._in = innodes
        self._out = outnodes

    input  = property(lambda self: self._in)
    output = property(lambda self: self._out)

    def __str__(self):
        s_in = ""
        s_out = ""
        for g in self._in:
            s_in += str(g)
        for g in self._out:
            s_out += str(g)
        s = "IN: " + s_in + " OUT: " + s_out
        return s

    def decode(self, mod_inputs, mod_type):
        # if there are multiple inputs they must be merged. Our choice for this is depthwise
        # concatenation. This is more computationally expensive than the other common solution,
        # summing. If input sizes don't match, they are downsampled to the smallest size
        if len(self._in) > 1:
            conn_inputs = []
            for layer in self._in[:]:
                try:
                    conn_inputs.append(mod_inputs[layer._id])
                except KeyError:
                    print(str(self))
                    raise KeyError
            # The below code uses MaxPooling layers to downsample convolutional layers
            # TODO add code to downsample dense layers
            if mod_type == 'CONV':
                conn_in_sizes = []
                for i in range(len(conn_inputs)):
                    conn_in_sizes.append(keras.backend.int_shape(conn_inputs[i])[1])
                min_size = min(conn_in_sizes)
                for i in range(len(conn_inputs)):
                    if conn_in_sizes[i] != min_size:
                        new_size = int(conn_in_sizes[i]/min_size)
                        conn_inputs[i] = keras.layers.MaxPool2D(new_size)(conn_inputs[i])
            x = keras.layers.Concatenate()(conn_inputs)
        else:
            x = mod_inputs[self._in[0]._id]
        for layer in self._out[:]:
            if layer._type == 'OUT':
                mod_inputs[-1] = x
            else:
                mod_inputs[layer._id] = layer.decode(x)
    def copy(self):
        return Connection(self._in.copy(), self._out.copy())

        ###############################################

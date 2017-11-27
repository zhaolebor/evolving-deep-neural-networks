# -*- coding: UTF-8 -*-
import random
from .config import Config
import keras
import math

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
            "_dropout": [0.1*i for i in range(7)],
            "_batch_norm": [True, False],
    }
    def __init__(self, id, numnodes, activation='relu', dropout=0.0, batch_norm=False, layertype='DENSE'):
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
        if r() < .2:
            self._max_pooling = random.choice(self.layer_params['_batch_norm'])
        elif r() < .2:
            self._dropout = random.choice(self.layer_params['_dropout'])
        elif r() < .2:
            self._size = random.choice(self.layer_params['_size'])

    def decode(self, x):
        inputdim = len(keras.backend.int_shape(x)[1:])
        if inputdim > 2:
            x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(self._size, activation=self._activation)(x)
        if self._batch_norm:
            x = keras.layers.BatchNormalization()(x)
        if self._dropout:
            x = keras.layers.Dropout(self._dropout)(x)
        return x

class ConvGene(LayerGene):

    layer_params = {
            "_size": [2**i for i in range(4, 9)],
            "_kernel_size": [1,3,5],
            "_activation": ['sigmoid','tanh','relu'],
            "_dropout": [.1*i for i in range(7)],
            "_padding": ['same','valid'],
            "_strides": [(1,1), (2,1), (1,2), (2,2)],
            "_max_pooling": list(range(3)),
            "_batch_norm": [True, False],
    }

    def __init__(self, id, numfilter, kernel_size=1, activation='relu', dropout=0.0, \
            padding='same', strides=(1,1), max_pooling=0, batch_norm=False, layertype='CONV'):
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
        r = random.random
        if r() < .2:
            self._max_pooling = random.choice(self.layer_params['_max_pooling'])
        elif r() < .2:
            self._dropout = random.choice(self.layer_params['_dropout'])
        elif r() < .2:
            self._kernel_size = random.choice(self.layer_params['_kernel_size'])
        elif r() < .2:
            self._size = random.choice(self.layer_params['_size'])

    def decode(self, x):
        inputdim = len(keras.backend.int_shape(x)[1:])
        if inputdim == 1:
            xval = keras.backend.int_shape(x)[1]
            dim1 = math.floor(math.sqrt(xval))
            while (xval%dim1 != 0):
                dim1 -= 1
            dim2 = xval/dim1
            x = keras.layers.Reshape((int(dim1),int(dim2),1))(x)
        x = keras.layers.Conv2D(self._size, self._kernel_size, strides=self._strides, \
                padding=self._padding, activation=self._activation)(x)
        if self._dropout:
            x = keras.layers.Dropout(self._dropout)(x)
        if self._max_pooling:
            for i in range(self._max_pooling):
                x = keras.layers.MaxPool2D()(x)
        return x



class ModuleGene(object):
    _id = 0
    def __init__(self, id, modspecies):
        """
        A module gene is a node which represents a multilayer component of
        a deep neural network.
        """
        if (id == None):
            self._id = self.__get_new_id()
        else:
            self._id = id
        self._type = 'MODULE'
        self._module = modspecies

    @classmethod
    def __get_new_id(cls):
        cls._id += 1
        return cls._id

    id = property(lambda self: self._id)
    type = property(lambda self: self._type)
    module = property(lambda self: self._module)

    def __str__(self):
        return "Module ID: %2d Species ID: %2d"%(self._id, self._module.id)

    def get_child(self, other):
        """
        Creates a new NodeGene randonly inheriting its attributes from
        parents.
        """
        assert(self._type == other._type)

        g = ModuleGene(self._id, random.choice(self._module, other._module))
        return g

    def copy(self):
        return ModuleGene(self._id, self._module)

    def set_module(self, modspecies):
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

    def decode(self, mod_inputs):
        if len(self._in) > 1:
            conn_inputs = []
            for mod in self._in:
                try:
                    conn_inputs.append(mod_inputs[mod._id])
                except KeyError:
                    print(str(self))
                    raise KeyError
            x = keras.layers.Add()(conn_inputs)
        else:
            x = mod_inputs[self._in[0]._id]
        for layer in self._out:
            if layer._type == 'OUT':
                mod_inputs[-1] = x
            else:
                mod_inputs[layer._id] = layer.decode(x)

        ###############################################

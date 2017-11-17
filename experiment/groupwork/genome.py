# -*- coding: UTF-8 -*-
import random
from .config import Config

class NodeGene(object):
    def __init__(self, id, nodetype):
        """
        A node gene defines the basic unit of the chromosome.
        """
        self._id = id
        self._type = nodetype

    id = property(lambda self: self._id)
    type = property(lambda self: self._type)

    def __str__(self):
        return "Node %2d %6s " \
                %(self._id, self._type)

    def get_child(self, other):
        """
        Creates a new NodeGene randonly inheriting its attributes from
        parents.
        """
        assert(self._type == other._type)

        ng = NodeGene(self._id, self._type,
                      random.choice((self._bias, other._bias)),
                      random.choice((self._response, other._response)),
                      self._activation_type)
        return ng

    def _mutate_bias(self):
        #self._bias += random.uniform(-1, 1) * Config.bias_mutation_power
        self._bias += random.gauss(0,1)*Config.bias_mutation_power
        if self._bias > Config.max_weight:
            self._bias = Config.max_weight
        elif self._bias < Config.min_weight:
            self._bias = Config.min_weight

    def copy(self):
        return NodeGene(self._id, self._type)

    def mutate(self):
        pass

class LayerGene(object):
    _id = 0
    def __init__(self, id, layertype, outputdim):
        """
        A layer gene is a node which represents a single Keras layer.
        """
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
        return "Layer %2d %6s %6s" \
                %(self._id, self._type, self._size)

    def get_child(self, other):
        """
        Creates a new LayerGene randonly inheriting its attributes from
        parents.
        """
        assert(self._type == other._type)

        g = LayerGene(self.__get_new_id, self._type, random.choice(self._size, other._size))
        return g

    def copy(self):
        return LayerGene(self.__get_new_id(), self._type, self._size)

    def mutate(self):
        raise NotImplementedError

class DenseGene(LayerGene):
    def __init__(self, id, numnodes, activation='relu', dropout=0.0, batch_norm=False \
            layertype='DENSE'):
        super(DenseGene, self).__init__(id, layertype, numnodes)
        self._activation = activation
        self._dropout = dropout
        self._batch_norm = batch_norm
        self.layer_params = {
            "_size": [2**i for i in range(4, int(math.log(256, 2)) + 1)],
            "_activation": ['sigmoid', 'tanh', 'relu']
            "_dropout": [0.0, 0.7]
            "_batch_norm": [True, False],
        }

    def get_child(self, other):
        assert(self._type == other._type)
        child_param = []
        for key in self.layer_params:
           child_param.append(random.choice(self.key, other.key))
        return DenseGene(self.__get_new_id(), child_param[0], child_param[1], child_param[2], \
                child_param[3])

    def copy(self):
        return DenseGene(self.__get_new_id(), self._size, self._activation, \
                self._dropout, self._batch_norm)

    def mutate(self):
        pass

class ConvGene(LayerGene):
    def __init__(self, id, numfilter, kernel_size=1, activation='relu', dropout=0.0, \
            padding='same', strides=(1,1), max_pooling=0, batch_norm=False, layertype='CONV'):
        super(DenseGene, self).__init__(id, layertype, numfilter)
        self._kernel_size = kernel_size
        self._activation = activation
        self._dropout = dropout
        self._padding = padding
        self._strides = strides
        self._max_pooling = max_pooling
        self._batch_norm = batch_norm
        self.layer_params = {
            "_size": [2**i for i in range(1, 10)],
            "_kernel_size": [1,3,5],
            "_activation": ['sigmoid','tanh','relu'],
            "_dropout": [(i if dropout else 0) for i in range(11)],
            "_padding": ['same','valid'],
            "_strides": [(1,1), (2,1), (1,2), (2,2)],
            "_max_pooling": list(range(3)),
            "_batch_norm": [True, False],
        }
    def get_child(self, other):
        assert(self._type == other._type)
        child_param = []
        for key in self.layer_params:
           child_param.append(random.choice(self.key, other.key))
        return ConvGene(self.__get_new_id(), child_param[0], child_param[1], child_param[2], \
                child_param[3], child_param[4], child_param[5], child_param[6], child_param[7])


    def copy(self):
        return ConvGene(self.__get_new_id(), self._size, self._activation, self._dropout, \
                self._padding, self._strides, self._max_pooling, self._batch_norm)

    def mutate(self):
        pass


class ModuleGene(object):
    def __init__(self, id, modtype, modspecies):
        """
        A module gene is a node which represents a multilayer component of
        a deep neural network.
        """
        self._id = id
        self._type = layertype
        self._module = modspecies

    id = property(lambda self: self._id)
    type = property(lambda self: self._type)
    module = property(lambda self: self._module)

    def __str__(self):
        return "Module %2d %6s " \
                %(self._id, self._type, self._module._species_id)

    def get_child(self, other):
        """
        Creates a new NodeGene randonly inheriting its attributes from
        parents.
        """
        assert(self._type == other._type)

        g = ModuleGene(self._id, self._type, random.choice(self._module, other._module))
        return g

    def copy(self):
        return ModuleGene(self._id, self._type, self._module)

    def mutate(self):
        pass


class Connection(object):
    #__global_innov_number = 0
    #__innovations = {} # A list of innovations.
    # Should it be global? Reset at every generation? Who knows?

    #@classmethod
    #def reset_innovations(cls):
    #    cls.__innovations = {}

    def __init__(self, innodes, outnodes, innov = None):
        self.__in = innodes
        self.__out = outnodes
        '''
        if innov is None:
            try:
                self.__innov_number = self.__innovations[self.key]
            except KeyError:
                self.__innov_number = self.__get_new_innov_number()
                self.__innovations[self.key] = self.__innov_number
        else:
            self.__innov_number = innov
        self.__out_node = None
        '''

    innodes  = property(lambda self: self.__in)
    outnodes = property(lambda self: self.__out)
    # Key for dictionaries, avoids two connections between the same nodes.
    # key = property(lambda self: (self.__in, self.__out))

    def mutate(self):
        r = random.random
        if r() < Config.prob_mutate_weight:
            self.__mutate_weight()
        if r() <  Config.prob_togglelink:
            self.enable()
        #TODO: Remove weight_replaced?
        #if r() < 0.001:
        #    self.__weight_replaced()
    '''
    @classmethod
    def __get_new_innov_number(cls):
        cls.__global_innov_number += 1
        return cls.__global_innov_number
    '''
    def __str__(self):
        s = "In %2d, Out %2d, Weight %+3.5f, " % \
            (self.__in, self.__out, self.__weight)
        if self.__enabled:
            s += "Enabled, "
        else:
            s += "Disabled, "
        return s + "Innov %d" % (self.__innov_number,)

    def __cmp__(self, other):
        return cmp(int(self.__innov_number), int(other.__innov_number))

    def __lt__(self, other):
        return int(self.__innov_number) <  int(other.__innov_number)

    def __gt__(self, other):
        return int(self.__innov_number) > int(other.__innov_number)

    def add_in(self, node_in):
        self.__in.append(node_in)

    def split(self, node_id):
        """
        Splits a connection, creating two new connections and
        disabling this one
        """
        self.__enabled = False
        new_conn1 = ConnectionGene(self.__in, node_id, 1.0, True)
        new_conn2 = ConnectionGene(node_id, self.__out, self.__weight, True)
        return new_conn1, new_conn2


    def copy(self):
        toReturn = ConnectionGene(self.__in, self.__out, self.__weight,
                              self.__enabled, self.__innov_number, self.__type)
        return toReturn

    def is_same_innov(self, cg):
        return self.__innov_number == cg.__innov_number

    def get_child(self, cg):
        # TODO: average both weights (Stanley, p. 38)
        return random.choice((self, cg)).copy()

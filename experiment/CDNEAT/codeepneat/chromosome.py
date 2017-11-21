import random, math
from .config import Config
from . import genome
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend

class Chromosome(object):
    """ A chromosome data type for co-evolving deep neural networks """
    _id = 0
    def __init__(self, parent1_id, parent2_id, gene_type):

        self._id = self.__get_new_id()

        # the type of ModuleGene or LayerGene the chromosome carries
        self._gene_type = gene_type
        # how many genes of the previous type the chromosome has
        self._genes = []

        self.fitness = None
        self.num_use = None
        self.species_id = None

        # my parents id: helps in tracking chromosome's genealogy
        self.parent1_id = parent1_id
        self.parent2_id = parent2_id

    genes      = property(lambda self: self._genes)
    id         = property(lambda self: self._id)

    @classmethod
    def __get_new_id(cls):
        cls._id += 1
        return cls._id

    def mutate(self):
        """ Mutates this chromosome """
        raise NotImplementedError

    def crossover(self, other):
        """ Crosses over parents' chromosomes and returns a child. """

        # This can't happen! Parents must belong to the same species.
        assert self.species_id == other.species_id, 'Different parents species ID: %d vs %d' \
                                                         % (self.species_id, other.species_id)

        if self.fitness > other.fitness:
            parent1 = self
            parent2 = other
        elif self.fitness == other.fitness:
            if len(self._genes) > len(other._genes):
                parent1 = other
                parent2 = self
            else:
                parent1 = self
                parent2 = other
        else:
            parent1 = other
            parent2 = self

        # creates a new child
        child = self.__class__(self._input, self.id, other.id, self._gene_type)

        child._inherit_genes(parent1, parent2)

        child.species_id = parent1.species_id

        return child

    def _inherit_genes(child, parent1, parent2):
        """ Applies the crossover operator. """
        raise NotImplementedError

    # compatibility function
    def distance(self, other):
        """ Returns the distance between this chromosome and the other. """
        if len(self._genes) > len(other._genes):
            chromo1 = self
            chromo2 = other
        else:
            chromo1 = other
            chromo2 = self

        return chomo1._genes-chromo2._genes

    def size(self):
        """ Defines chromosome 'complexity': number of hidden nodes plus
            number of enabled connections (bias is not considered)
        """
        return len(self._genes)

    def __cmp__(self, other):
        return cmp(self.fitness, other.fitness)

    def __lt__(self, other):
        return self.fitness <  other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __str__(self):
        s = "Genes:"
        for ng in self._genes:
            s += "\n\t" + str(ng)
        return s

class BlueprintChromo(Chromosome):
    """
    A chromosome for deep neural network blueprints.
    """
    def __init__(self, parent1_id, parent2_id, module_pop, active_params={}, gene_type='MODULE'):
        super(BlueprintChromo, self).__init__(parent1_id, parent2_id, gene_type)
        self._species_indiv = {}
        self._all_params = {
                "learning_rate": [.0001, .1],
                "momentum": [.68,.99],
                "hue_shift": [0, 45],
                "sv_shift": [0, .5],
                "sv_scale": [0, .5],
                "cropped_image_size": [26,32],
                "spatial_scaling": [0,0.3],
                "horizontal_flips": [True, False],
                "variance_norm": [True, False],
                "nesterov_acc": [True, False],
                }
        self._active_params = active_params
        self._module_pop = module_pop

    def __get_species_indiv(self):
        for i in range(len(self._genes)):
            if not (self._module_pop.has_species(self._genes[i].module)):
                self._genes[i].set_module(self._module_pop.get_species())
        for g in self._genes:
            try:
                self._species_indiv[g.module.id] = g.module.get_indiv()
            except KeyError:
                pass

    def decode(self, inputs):
        self.__get_species_indiv()
        next = inputs
        for g in self._genes:
            mod = self._species_indiv[g.module.id].decode(next)
            next = mod(next)
        return next

    def distance(self, other):
        dist = 0
        if len(self._genes) > len(other._genes):
            chromo1 = self
            chromo2 = other
        else:
            chromo1 = other
            chromo2 = self
        dist += (len(chromo1._genes)-len(chromo2._genes))*Config.excess_coefficient
        chromo1_species = []
        chromo2_species = []
        for g in chromo1._genes:
            chromo1_species.append(g.module.id)
        for g in chromo2._genes:
            chromo2_species.append(g.module.id)
        for id in chromo1_species:
            if id in chromo2_species:
                chromo2_species.remove(id)
            else:
                dist += Config.disjoint_coefficient
        return dist

    def _inherit_genes(child, parent1, parent2):
        """ Applies the crossover operator. """
        assert(parent1.fitness >= parent2.fitness)

        # Crossover layer genes
        for i, g1 in enumerate(parent1._genes):
            try:
                # matching node genes: randomly selects the neuron's bias and response
                child._genes.append(g1.get_child(parent2._genes[i]))
            except IndexError:
                # copies extra genes from the fittest parent
                child._genes.append(g1.copy())
            except TypeError:
                # copies disjoint genes from the fittest parent
                child._genes.append(g1.copy())

    def mutate(self):
        """ Mutates this chromosome """

        r = random.random
        if r() < Config.prob_addmodule:
            self._mutate_add_module()
        else:
            for param in list(self._active_params.keys):
                if r() < 0.5:
                    self._active_params[param] = random.choice(self._all_params[param])
        return self

    def _mutate_add_module(self):
        """ Adds a module to the BluePrintChromo"""
        ind = random.randint(0,len(self._genes))
        module = self._module_pop.get_species()
        self._genes.insert(ind, genome.ModuleGene(None, module))

    @classmethod
    def create_initial(cls, module_pop):
        c = cls(None, None, module_pop)
        c._genes.append(genome.ModuleGene(None, module_pop.get_species()))
        c._genes.append(genome.ModuleGene(None, module_pop.get_species()))
        return c


class ModuleChromo(Chromosome):
    """
    A chromosome for deep neural network "modules" which consist of a small number of layers and
    their associated hyperparameters.
    """
    def __init__(self, parent1_id, parent2_id, gene_type='DENSE'):
        super(ModuleChromo, self).__init__(parent1_id, parent2_id, gene_type)
        self._connections = []

    def decode(self, inputs):
        """
        Produces Keras Model of this module
        """
        inputdim = backend.int_shape(inputs)[1:]
        inputlayer = Input(shape=inputdim)
        mod_inputs = {0: inputlayer}
        for conn in self._connections:
            conn.decode(mod_inputs)
        mod = Model(inputs=inputlayer, outputs=mod_inputs[-1])
        return mod

    def distance(self, other):
        dist = 0
        if len(self._genes) > len(other._genes):
            chromo1 = self
            chromo2 = other
        else:
            chromo1 = other
            chromo2 = self
        dist += (len(chromo1._connections)-len(chromo2._connections))*Config.excess_coefficient
        if (chromo1._gene_type != chromo2._gene_type):
            return (dist+1000)
        else:
            for i, conn in enumerate(chromo2._connections):
                for j in range(len(chromo1._connections[i]._in)):
                    if chromo1._connections[i]._in[j] not in list(conn._in):
                        dist += Config.connection_coefficient
        return dist

    def _inherit_genes(child, parent1, parent2):
        """ Applies the crossover operator. """
        assert(parent1.fitness >= parent2.fitness)

        # Crossover layer genes
        for i, g1 in enumerate(parent1._genes):
            try:
                # matching node genes: randomly selects the neuron's bias and response
                child._genes.append(g1.get_child(parent2._genes[i]))
            except IndexError:
                # copies extra genes from the fittest parent
                child._genes.append(g1.copy())
            except TypeError:
                # copies disjoint genes from the fittest parent
                child._genes.append(g1.copy())
        child._connections = parent1._connections.copy()
        for conn in child._connections:
            for inlayer in conn._in:
                if inlayer.type != 'IN':
                    ind = parent1._genes.index(inlayer)
                    inlayer = child._genes(ind)
            for outlayer in conn._out:
                if outlayer.type != 'OUT':
                    ind = parent1._genes.index(outlayer)
                    outlayer = child._genes(ind)

    def mutate(self):
        """ Mutates this chromosome """

        r = random.random
        if r() < Config.prob_addlayer:
            self._mutate_add_layer()

        else:
            for g in self._genes:
                g.mutate() # mutate layer params

        return self

    def _mutate_add_layer(self):
        r = random.random
        if self._gene_type == 'CONV':
            if r() < Config.prob_addconv:
                ng = genome.ConvGene(None,32)
                self._genes.append(ng)
        else:
            ng = genome.DenseGene(None,128)
            self._genes.append(ng)
        n = genome.LayerGene(0, 'IN', 0)
        x = genome.LayerGene(-1, 'OUT', 0)
        conns = self._connections.copy()
        conns.insert(0, genome.Connection([n], []))
        inlayers = random.choice(conns)
        inlayers._out.append(ng)
        if len(inlayers._in) == 1 and inlayers._in[0].type == 'IN':
            self._connections[0]._in.pop(0)
            self._connections[0]._in.append(ng)
            self._connections.insert(0,inlayers)
        conns.append(genome.Connection([], [x]))
        output = random.choice(conns[conns.index(inlayers):])
        if ng in output._in:
            if len(output._in) == 1:
                output = random.choice(conns[cons.index(inlayers)+1:])
            else:
                inlayers2 = output._in.copy()
                inlayers2.remove(ng)
                self._connections.insert(output, genome.Connection(inlayers2, [ng]))
                return
        output._in.append(ng)
        if len(output._out) == 1 and output._out[0].type == 'OUT':
            self._connections[-1]._out.pop(0)
            self._connections[-1]._out.append(ng)
            self._connections.append(output)


    @classmethod
    def create_initial(cls):
        c = cls(None,None)
        n = genome.LayerGene(0, 'IN', 0)
        x = genome.LayerGene(-1, 'OUT', 0)
        if Config.conv and random.random() > Config.prob_addconv:
            c._gene_type = 'CONV'
            g = genome.ConvGene(None, 32)
            c._genes.append(g)
            c._connections.append(genome.Connection([n],[g]))
            c._connections.append(genome.Connection([g],[x]))
        else:
            g = genome.DenseGene(None, 128)
            c._genes.append(g)
            c._connections.append(genome.Connection([n],[g]))
            c._connections.append(genome.Connection([g],[x]))
        return c


import random, math
from .config import Config
from . import hierarchy
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

    def decode(self):
        """ Creates Neural Network from genome """
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
        child = _create_child(parent1, parent2)

        child.species_id = parent1.species_id

        return child

    def _inherit_genes(child, parent1, parent2):
        """ Applies the crossover operator. """
        raise NotImplementedError

    def _create_child(self, other):
        return self.__class__(self.id, other.id, self._gene_type)

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

    def copy(self):
        """
        Returns shallow copy of chromosome
        """
        return self.__class__(self.id, other.id, self._gene_type)

    def __cmp__(self, other):
        return cmp(self.fitness, other.fitness)

    def __lt__(self, other):
        return self.fitness <  other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __str__(self):
        s = "ID: "+str(self._id)+" Species: "+str(self.species_id)+"\nGenes:"
        for ng in self._genes:
            s += "\n\t" + str(ng)
        return s

class BlueprintChromo(Chromosome):
    """
    A chromosome for deep neural network blueprints.
    """
    def __init__(self, parent1_id, parent2_id, active_params={}, gene_type='MODULE', genes=None):
        super(BlueprintChromo, self).__init__(parent1_id, parent2_id, gene_type)
        if genes is not None:
            self._genes = genes
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

    def _create_child(self, parent1, parent2):

        child = BlueprintChromo(parent1.id, parent2.id, self._active_params)
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
        # Insure inherited module genes point to actual species
        for g in child._genes.copy():
            if not (child._module_pop.has_species(g.module)):
                new_species = child._module_pop.get_species()
                fails = 0
                while True:
                    try:
                        g.set_module(new_species)
                        break
                    except TypeError:
                        fails += 1
                        new_species = child._module_pop.get_species()
                        if fails > 500:
                            child._genes.remove(g)
                            break



    def __get_species_indiv(self):
        # This method chooses an individule for each unique Module Species Gene
        for g in self._genes:
            # Check that each gene actually points to an existing species
            if not self._module_pop.has_species(g.module):
                raise ValueError('Missing species at decode: '+str(g.module))
        self._species_indiv = {}
        for g in self._genes:
            if g.module.id not in self._species_indiv:
                self._species_indiv[g.module.id] = random.choice(g.module.members)

    def decode(self, inputs):
        # create keras Model from component modules
        self.__get_species_indiv()
        next = inputs
        for g in self._genes:
            mod = self._species_indiv[g.module.id].decode(next)
            next = mod(next)
        return next

    def distance(self, other):
        # measure distance between two blueprint chromosomes for speciation purpose
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
        # Insure inherited module genes point to actual species
        for g in child._genes.copy():
            if not (child._module_pop.has_species(g.module)):
                new_species = child._module_pop.get_species()
                fails = 0
                while True:
                    try:
                        g.set_module(new_species)
                        break
                    except TypeError:
                        fails += 1
                        new_species = child._module_pop.get_species()
                        if fails > 500:
                            child._genes.remove(g)
                            break


    def mutate(self):
        """ Mutates this chromosome """

        r = random.random
        if r() < Config.prob_addmodule:
            self._mutate_add_module()
        elif len(self._active_params) > 0:
            for param in list(self._active_params.keys):
                if r() < 0.5:
                    self._active_params[param] = random.choice(self._all_params[param])
        g = random.choice(self._genes)
        if r() < Config.prob_switchmodule:
            new_species = self._module_pop.get_species()
            fails = 0
            while True:
                try:
                    g.set_module(new_species)
                    break
                except TypeError:
                    fails += 1
                    new_species = self._module_pop.get_species()
                    if fails > 500:
                        self._genes.remove(g)
                        break
        return self

    def _mutate_add_module(self):
        """ Adds a module to the BluePrintChromo"""
        ind = random.randint(0,len(self._genes))
        if Config.conv and ind > 0:
            valid_mod = False
            while not valid_mod:
                module = self._module_pop.get_species()
                mod_type = module.members[0].type
                if self._genes[ind-1].type != 'CONV' and mod_type == 'CONV':
                    valid_mod = False
                else:
                    valid_mod = True
        else:
            module = self._module_pop.get_species()
            mod_type = module.members[0].type
        self._genes.insert(ind, genome.ModuleGene(None, module, mod_type))

    def copy(self):
        """ NOT TRUE COPY METHOD, returns pointer to self with valid species pointers"""
        for g in self._genes:
            if not (self._module_pop.has_species(g.module)):
                new_species = self._module_pop.get_species()
                fails = 0
                while True:
                    try:
                        g.set_module(new_species)
                        break
                    except TypeError:
                        fails += 1
                        new_species = self._module_pop.get_species()
                        if fails > 500:
                            self._genes.remove(g)
                            break
        return self

    @classmethod
    def create_initial(cls, module_pop):
        c = cls(None, None, module_pop)
        n = random.randrange(2,5)
        mod = module_pop.get_species()
        mod_type = mod.members[0].type
        c._genes.append(genome.ModuleGene(None, mod, mod_type))
        for i in range(1,n):
            c._genes.append(genome.ModuleGene(None, mod, mod_type))
        if Config.conv and Config.prob_addconv < 1:
            conv_ind = []
            for i in range(len(c._genes)):
                if c._genes[i].type == 'CONV':
                    conv_ind.append(i)
            conv_mods = []
            for j in range(len(conv_ind)-1, 0, -1):
                mod = c._genes.pop(conv_ind[j])
                conv_mods.append(mod)
            for k in range(len(conv_mods)):
                c._genes.insert(0,conv_mods.pop())
        return c


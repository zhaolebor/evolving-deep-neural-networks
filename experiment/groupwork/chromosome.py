import random, math
from .config import Config
from . import genome

# Temporary workaround - default settings
#node_gene_type = genome.NodeGene
conn_gene_type = genome.ConnectionGene

class Chromosome(object):
    """ A chromosome for general recurrent neural networks. """
    _id = 0
    def __init__(self, input, output, parent1_id, parent2_id, gene_type):

        self._id = self.__get_new_id()
        self._input  = input
        self._output = output

        # the type of NodeGene and ConnectionGene the chromosome carries
        self._gene_type = gene_type
        # how many genes of the previous type the chromosome has
        self._connections = {} # dictionary of connection genes
        self._genes = []

        self.fitness = None
        self.species_id = None

        # my parents id: helps in tracking chromosome's genealogy
        self.parent1_id = parent1_id
        self.parent2_id = parent2_id

    conn       = property(lambda self: list(self._connections.values()))
    genes      = property(lambda self: self._genes)
    input      = property(lambda self: self._input)
    output     = property(lambda self: self._output)
    id         = property(lambda self: self._id)

    @classmethod
    def __get_new_id(cls):
        cls._id += 1
        return cls._id

    def mutate(self):
        """ Mutates this chromosome """

        r = random.random
        if r() < Config.prob_addnode:
            self._mutate_add_node()

        elif r() < Config.prob_addconn:
            self._mutate_add_connection()

        else:
            for cg in list(self._connection_genes.values()):
                cg.mutate() # mutate weights
            for ng in self._node_genes[self._input_nodes:]:
                ng.mutate() # mutate bias, response, and etc...

        return self


    def crossover(self, other):
        """ Crosses over parents' chromosomes and returns a child. """

        # This can't happen! Parents must belong to the same species.
        assert self.species_id == other.species_id, 'Different parents species ID: %d vs %d' \
                                                         % (self.species_id, other.species_id)

        # TODO: if they're of equal fitnesses, choose the shortest
        if self.fitness > other.fitness:
            parent1 = self
            parent2 = other
        else:
            parent1 = other
            parent2 = self

        # creates a new child
        child = self.__class__(self._input, self._output, self.id, other.id, self._node_gene_type)

        child._inherit_genes(parent1, parent2)

        child.species_id = parent1.species_id

        return child

    def _inherit_genes(child, parent1, parent2):
        """ Applies the crossover operator. """
        assert(parent1.fitness >= parent2.fitness)

        # Crossover connection genes
        for cg1 in list(parent1._connections.values()):
            try:
                cg2 = parent2._connections[cg1.key]
            except KeyError:
                # Copy excess or disjoint genes from the fittest parent
                child._connections[cg1.key] = cg1.copy()
            else:
                if cg2.is_same_innov(cg1): # Always true for *global* INs
                    # Homologous gene found
                    new_gene = cg1.get_child(cg2)
                    #new_gene.enable() # avoids disconnected neurons
                else:
                    new_gene = cg1.copy()
                child._connections[new_gene.key] = new_gene

        # Crossover node genes
        for i, ng1 in enumerate(parent1._genes):
            try:
                # matching node genes: randomly selects the neuron's bias and response
                child._genes.append(ng1.get_child(parent2._genes[i]))
            except IndexError:
                # copies extra genes from the fittest parent
                child._genes.append(ng1.copy())


    def _mutate_add_node(self):
        throw NotImplementedError

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

class FFChromosome(Chromosome):
    """ A chromosome for feedforward neural networks. Feedforward
        topologies are a particular case of Recurrent NNs.
    """
    def __init__(self, parent1_id, parent2_id, node_gene_type, conn_gene_type):
        super(FFChromosome, self).__init__(parent1_id, parent2_id, node_gene_type, conn_gene_type)
        self.__node_order = [] # hidden node order (for feedforward networks)

    node_order = property(lambda self: self.__node_order)

    def _inherit_genes(child, parent1, parent2):
        super(FFChromosome, child)._inherit_genes(parent1, parent2)

        child.__node_order = parent1.__node_order[:]

        assert(len(child.__node_order) == len([n for n in child.node_genes if n.type == 'HIDDEN']))

    def _mutate_add_node(self):
        ng, split_conn = super(FFChromosome, self)._mutate_add_node()
        # Add node to node order list: after the presynaptic node of the split connection
        # and before the postsynaptic node of the split connection
        if self._node_genes[split_conn.innodeid - 1].type == 'HIDDEN':
            mini = self.__node_order.index(split_conn.innodeid) + 1
        else:
            # Presynaptic node is an input node, not hidden node
            mini = 0
        if self._node_genes[split_conn.outnodeid - 1].type == 'HIDDEN':
            maxi = self.__node_order.index(split_conn.outnodeid)
        else:
            # Postsynaptic node is an output node, not hidden node
            maxi = len(self.__node_order)
        self.__node_order.insert(random.randint(mini, maxi), ng.id)
        assert(len(self.__node_order) == len([n for n in self.node_genes if n.type == 'HIDDEN']))
        return (ng, split_conn)

    def _mutate_add_connection(self):
        # Only for feedforwad networks
        num_hidden = len(self.__node_order)
        num_output = len(self._node_genes) - self._input_nodes - num_hidden

        total_possible_conns = (num_hidden+num_output)*(self._input_nodes+num_hidden) - \
            sum(range(num_hidden+1))

        remaining_conns = total_possible_conns - len(self._connection_genes)
        # Check if new connection can be added:
        if remaining_conns > 0:
            n = random.randint(0, remaining_conns - 1)
            count = 0
            # Count connections
            for in_node in (self._node_genes[:self._input_nodes] + self._node_genes[-num_hidden:]):
                for out_node in self._node_genes[self._input_nodes:]:
                    if (in_node.id, out_node.id) not in list(self._connection_genes.keys()) and \
                        self.__is_connection_feedforward(in_node, out_node):
                        # Free connection
                        if count == n: # Connection to create
                            #weight = random.uniform(-Config.random_range, Config.random_range)
                            weight = random.gauss(0,1)
                            cg = self._conn_gene_type(in_node.id, out_node.id, weight, True)
                            self._connection_genes[cg.key] = cg
                            return
                        else:
                            count += 1

    def __is_connection_feedforward(self, in_node, out_node):
        return in_node.type == 'INPUT' or out_node.type == 'OUTPUT' or \
            self.__node_order.index(in_node.id) < self.__node_order.index(out_node.id)

    def add_hidden_nodes(self, num_hidden):
        id = len(self._node_genes)+1
        for i in range(num_hidden):
            node_gene = self._node_gene_type(id,
                                          nodetype = 'HIDDEN',
                                          activation_type = Config.nn_activation)
            self._node_genes.append(node_gene)
            self.__node_order.append(node_gene.id)
            id += 1
            # Connect all input nodes to it
            for pre in self._node_genes[:self._input_nodes]:
                weight = random.gauss(0, Config.weight_stdev)
                cg = self._conn_gene_type(pre.id, node_gene.id, weight, True)
                self._connection_genes[cg.key] = cg
                assert self.__is_connection_feedforward(pre, node_gene)
            # Connect all previous hidden nodes to it
            for pre_id in self.__node_order[:-1]:
                assert pre_id != node_gene.id
                weight = random.gauss(0, Config.weight_stdev)
                cg = self._conn_gene_type(pre_id, node_gene.id, weight, True)
                self._connection_genes[cg.key] = cg
            # Connect it to all output nodes
            for post in self._node_genes[self._input_nodes:(self._input_nodes + self._output_nodes)]:
                assert post.type == 'OUTPUT'
                weight = random.gauss(0, Config.weight_stdev)
                cg = self._conn_gene_type(node_gene.id, post.id, weight, True)
                self._connection_genes[cg.key] = cg
                assert self.__is_connection_feedforward(node_gene, post)

    def __str__(self):
        s = super(FFChromosome, self).__str__()
        s += '\nNode order: ' + str(self.__node_order)
        return s

def BlueprintChromo(Chromosome):
    """
    A chromosome for deep neural network blueprints.
    """
    def __init__(self, input, output, parent1_id, parent2_id, node_order=[], gene_type='MODULE'):
        super(BluePrintChromo, self).__init__(input, output, parent1_id, parent2_id, gene_type)
        self.__node_order = node_order # module order

def ModuleChromo(Chromosome):
    """
    A chromosome for deep neural network "modules" which consist of a small number of layers and
    their associated hyperparameters.
    """
    def __init__(self, input, output, parent1_id, parent2_id, gene_type='DENSE'):
        super(Module, self).__init__(parent1_id, parent2_id, node_gene_type, conn_gene_type)
        self.__node_order = [] # hidden node order (for feedforward networks)

if __name__ == '__main__':
    # Example
    from . import visualize
    # define some attributes
    node_gene_type = genome.NodeGene         # standard neuron model
    conn_gene_type = genome.ConnectionGene   # and connection link
    Config.nn_activation = 'exp'             # activation function
    Config.weight_stdev = 0.9                # weights distribution

    Config.input_nodes = 2                   # number of inputs
    Config.output_nodes = 1                  # number of outputs

    # creates a chromosome for recurrent networks
    #c1 = Chromosome.create_fully_connected()

    # creates a chromosome for feedforward networks
    c2 = FFChromosome.create_fully_connected()
    # add two hidden nodes
    c2.add_hidden_nodes(2)
    # apply some mutations
    #c2._mutate_add_node()
    #c2._mutate_add_connection()

    # check the result
    #visualize.draw_net(c1) # for recurrent nets
    visualize.draw_ff(c2)   # for feedforward nets
    # print the chromosome
    print(c2)

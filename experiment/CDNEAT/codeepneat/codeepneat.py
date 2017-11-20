from . import population
from . import species
from . import chromosome
from . import genome
from .config import Config
from random import random
import keras

def evaluate(blueprint_pop, module_pop, num_networks, f):
    for module in module_pop:
        module.fitness = 0
        module.num_use = 0
    for blueprint in blueprint_pop:
        blueprint.fitness = 0
        blueprint.num_use = 0
    networks = []
    bps = []
    for i in range(num_networks):
        bp = random.choice(blueprint_pop)
        bps.append(bp)
        model = bp.decode(Config.input_nodes)
        predictions = keras.layers.Dense(Config.output_nodes, activation='softmax')(model)
        net = keras.models.Model(input_nodes, predictions)
        net.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
        networks.append(net)
        bp.num_use += 1
        for module in list(bp._species_indiv.values()):
            module.num_use += 1
    for i in range(num_networks):
        fit = f(networks[i])
        bps[i].fitness += fit
        for module in list(bp._species_indiv.values()):
            module.fitness += fit




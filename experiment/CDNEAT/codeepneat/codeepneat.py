from . import population
from . import species
from . import chromosome
from . import genome
from .config import Config
import random
import keras

def produce_net(bp):
    inputs = keras.layers.Input(Config.input_nodes, name='input')
    x = bp.decode(inputs)
    x_dim = len(keras.backend.int_shape(x)[1:])
    if x_dim > 2:
        x = keras.layers.Flatten()(x)
    predictions = keras.layers.Dense(Config.output_nodes, activation='softmax')(x)
    net = keras.models.Model(inputs=inputs, outputs=predictions)
    net.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    return net


def evaluate(blueprint_pop, module_pop, num_networks, f, data):
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
        net = produce_net(bp)
        networks.append(net)
        bp.num_use += 1
        for module in list(bp._species_indiv.values()):
            module.num_use += 1
    for i in range(num_networks):
        fit = f(networks[i], data)
        bps[i].fitness += fit
        for module in list(bp._species_indiv.values()):
            module.fitness += fit

def epoch(n, pop1, pop2, num_networks, f, data, save_best, name='', report=True):
    for g in range(n):
        evaluate(pop1, pop2, num_networks, f, data)
        n = pop1.epoch(g, report, save_best, name)
        m = pop2.epoch(g, report, save_best, name)
        if n < 0 or m < 0:
            break


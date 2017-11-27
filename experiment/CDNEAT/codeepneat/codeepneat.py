from . import population
from . import species
from . import chromosome
from . import genome
import time
from .config import Config
import pickle as pickle
import random
import h5py
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
    best_model = None
    best_fit = 0
    '''
    for i in range(num_networks):
        bp = random.choice(blueprint_pop)
        bps.append(bp)
        net = produce_net(bp)
        networks.append(net)
        bp.num_use += 1
        for module in list(bp._species_indiv.values()):
            module.num_use += 1
    '''
    for i in range(num_networks):
        bp = random.choice(blueprint_pop)
        net = produce_net(bp)
        bp.num_use += 1
        for module in list(bp._species_indiv.values()):
            module.num_use += 1
        print('Network '+ str(i))
        fit = f(net, data)
        print()
        print('Network '+ str(i) + ' Fitness: ' + str(fit))
        if fit > best_fit:
          best_fit = fit
          best_model = net
        bp.fitness += fit
        for module in list(bp._species_indiv.values()):
            module.fitness += fit
    '''
    for module in module_pop:
        print(str(module._id) + ' ' + str(module.num_use) + ' ' + str(module.fitness))
    '''
    for module in module_pop:
        if module.num_use == 0:
            module.fitness = .7
        else:
            module.fitness /= module.num_use
    for blueprint in blueprint_pop:
        if blueprint.num_use == 0:
            blueprint.fitness = .7
        else:
          blueprint.fitness /= blueprint.num_use
    return best_model

def epoch(n, pop1, pop2, num_networks, f, data, save_best, name='', report=True):
    for g in range(n):
        print("-----Generation "+str(g)+"--------")
        print_populations(pop1, pop2)
        best_model = evaluate(pop1, pop2, num_networks, f, data)
        print("-----Blueprints----------")
        j = pop1.epoch(g, report=report, save_best=False, name=name)
        print("-----Modules-----------")
        k = pop2.epoch(g, report=report, save_best=True, name=name)
        if save_best:
            filename = "best_model_" + str(g)
            if name != "":
                filename = name + "_" + filename
            best_model.save(filename)
        if j < 0 or k < 0:
            break

def print_populations(bp_pop, mod_pop):
  for bp in bp_pop:
    print(str(bp))
  for mod in mod_pop:
    print(str(mod))

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


def evaluate(blueprint_pop, module_pop, num_networks, f, data, debug=None):
    # Reset fitnesses for both populations
    for module in module_pop:
        module.fitness = 0
        module.num_use = 0
    for blueprint in blueprint_pop:
        blueprint.fitness = 0
        blueprint.num_use = 0
    # Generate a new set of networks from blueprints
    networks = []
    # Track the best model and its accuracy
    best_model = None
    best_fit = 0
    avg_fit = 0
    # Evaluate a given number of networks
    for i in range(num_networks):
        # Choose a random blue print
        bp = random.choice(blueprint_pop)
        ## Display chosen blue print for debugging
        if debug is not None:
            s = '\n ------------- Evaluating --------------- \n'
            s += 'Network ' + str(i) + '\n' + 'Blueprint: ' + str(bp) + '\n'
            debug.write(s)
        # Produce a Keras Model from the chosen blueprint chromosome
        net = produce_net(bp)
        ## Display the Modules the blueprint chose to include for debugging
        if debug is not None:
            s = ""
            for module in list(bp._species_indiv.values()):
                s += 'Module: ' + str(module) + '\n'
            debug.write(s)
        # Record that blueprint and module have been used in network
        bp.num_use += 1
        for module in list(bp._species_indiv.values()):
            module.num_use += 1
        print('Network '+ str(i))
        # Get fitness of created network
        fit = f(net, data)
        avg_fit += fit
        print()
        print('Network '+ str(i) + ' Fitness: ' + str(fit))
        # Record this fitness and model if its the best fitness of this generation
        if fit > best_fit:
          best_fit = fit
          best_model = net
        # Attribute fitness to blueprint and modules
        bp.fitness += fit
        for module in list(bp._species_indiv.values()):
            module.fitness += fit
    avg_fit /= num_networks
    # Set fitness of modules as average of attributed fitnesses
    # Set unused modules and blueprints at the average network fitness,
    # (plus .01 to encourage their survival to be used in the future)
    for module in module_pop:
        if module.num_use == 0:
            if debug is not None:
                debug.write('Unused module ' + str(module.id) + '\n')
            module.fitness = avg_fit + .01
        else:
            module.fitness /= module.num_use
    for blueprint in blueprint_pop:
        if blueprint.num_use == 0:
            if debug is not None:
                debug.write('Unused module ' + str(module.id) + '\n')
            blueprint.fitness = avg_fit + .01
        else:
          blueprint.fitness /= blueprint.num_use
    # return the highest performing single network
    return best_model

def epoch(n, pop1, pop2, num_networks, f, data, save_best, name='', report=True, debug=None):
    try:
        for g in range(n):
            print('-----Generation '+str(g)+'--------')
            if debug is not None:
                print_populations(pop1, pop2, debug)
            best_model = evaluate(pop1, pop2, num_networks, f, data, debug)
            print('-----Modules-----------')
            k = pop2.epoch(g, report=report, save_best=True, name=name+'_m')
            print('-----Blueprints----------')
            j = pop1.epoch(g, report=report, save_best=False, name=name)
            if save_best:
                filename = 'best_model_' + str(g)
                if name != '':
                    filename = name + '_' + filename
                best_model.save(filename)
            if j < 0 or k < 0:
                break
    except Exception as err:
        if debug is not None:
            debug.write('ERROR\n'+str(err.args))
            debug.close()
        raise err

def print_populations(bp_pop, mod_pop, debug):
    debug.write('\n ----------------- Blueprint Population --------------------- \n')
    for bp in bp_pop:
        debug.write(str(bp)+'\n')
    debug.write('\n ------------------ Module Population ----------------------- \n')
    for mod in mod_pop:
        debug.write(str(mod)+'\n')


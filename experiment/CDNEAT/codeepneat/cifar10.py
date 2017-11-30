"""
Python script to run an CoDeepNeat evolution for CIFAR 10
"""

def fitness(network, data):
        network.fit(data[0], data[1],  epochs=5)
        loss, acc = network.evaluate(data[2], data[3])
        return acc

def evolve(n):
    config.load('configCIFAR10')
    # Create 2 separate populations (size is now defined explicitly, but config file can still be used)
    module_pop = population.Population(10, chromosome.ModuleChromo)
    # As the top hierarchical level, the blueprint population needs to be able to see the module population
    blueprint_pop = population.Population(5, chromosome.BlueprintChromo, module_pop)
    # Most of the actual evolving is now handled outside of the population, by CoDeepNEAT
    # Instead of requiring the user to overwrite the evaluation function, CoDeepNEAT evaluates the populations itself
    # it simply requires a fitness function for the networks it creates passed in as an argument.
    codeepneat.epoch(n, blueprint_pop, module_pop, 10, fitness, data, save_best=True, name='CIFAR10')
    # It will still stop if fitness surpasses the max_fitness_threshold in config file
    # Plots the evolution of the best/average fitness
    visualize.plot_stats(module_pop.stats, name="CIFAR10mod_")
    visualize.plot_stats(blueprint_pop.stats, name="CIFAR10bp_")
    # Visualizes speciation\n",
    # visualize.plot_species(module_pop.species_log, name="NMISTmod_")
    # visualize.plot_species(blueprint_pop.species_log, name="NMISTbp_")

def main():
    evolve(10)

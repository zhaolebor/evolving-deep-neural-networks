import biggles
import pydot
import random
from keras.utils import plot_model

def draw_net(model, id=''):
    ''' Receives a chromosome and draws a neural network with arbitrary topology. '''
    plot_model(model, to_file='phenotype'+id+'.svg')


def plot_stats(stats, name=""):
    ''' 
    Plots the population's average and best fitness. 
    Lisa Meeden added a name parameter for handling multiple visualizations
    in co-evolution.
    '''
    generation = [i for i in range(len(stats[0]))]
    
    fitness = [fit for fit in stats[0]]
    avg_pop = [avg for avg in stats[1]]
    
    plot = biggles.FramedPlot()
    plot.title = "Pop. avg and best fitness"
    plot.xlabel = r"Generations"
    plot.ylabel = r"Fitness"
    
    plot.add(biggles.Curve(generation, fitness, color="red"))
    plot.add(biggles.Curve(generation, avg_pop, color="blue"))

    #plot.show() # X11
    plot.write_img(600, 300, name+'avg_fitness.svg')
    # width and height doesn't seem to affect the output! 


def plot_species(species_log, name=""):
    ''' 
    Visualizes speciation throughout evolution. 
    Lisa Meeden added a name parameter for handling multiple visualizations
    in co-evolution.
    '''
    plot = biggles.FramedPlot()
    plot.title = "Speciation"
    plot.ylabel = r"Size per Species"
    plot.xlabel = r"Generations"
    generation = [i for i in range(len(species_log))]

    species = []
    curves = []

    for gen in range(len(generation)):
        for j in range(len(species_log), 0, -1):
            try:
                species.append(species_log[-j][gen] + sum(species_log[-j][:gen]))
            except IndexError:
                species.append(sum(species_log[-j][:gen]))
        curves.append(species)
        species = []

    s1 = biggles.Curve(generation, curves[0])

    plot.add(s1)
    plot.add(biggles.FillBetween(generation, [0]*len(generation), generation, curves[0], color=random.randint(0,90000)))

    for i in range(1, len(curves)):
        c = biggles.Curve(generation, curves[i])
        plot.add(c)
        plot.add(biggles.FillBetween(generation, curves[i-1], generation, curves[i], color=random.randint(0,90000)))
    plot.write_img(1024, 800, name+'speciation.svg')

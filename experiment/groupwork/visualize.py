import biggles
import pydot
import random

def draw_net(chromosome, id=''):
    ''' Receives a chromosome and draws a neural network with arbitrary topology. '''
    output = 'digraph G {\n  node [shape=circle, fontsize=9, height=0.2, width=0.2]'

    # subgraph for inputs and outputs
    output += '\n  subgraph cluster_inputs { \n  node [style=filled, shape=box] \n    color=white'
    for ng in chromosome.node_genes:
        if ng.type== 'INPUT':
            output += '\n    '+str(ng.id)
    output += '\n  }'

    output += '\n  subgraph cluster_outputs { \n    node [style=filled, color=lightblue] \n    color=white'
    for ng in chromosome.node_genes:
        if ng.type== 'OUTPUT':
            output += '\n    '+str(ng.id)
    output += '\n  }'
    # topology
    for cg in chromosome.conn_genes:
        output += '\n  '+str(cg.innodeid)+' -> '+str(cg.outnodeid)
        if cg.enabled is False:
            output += ' [style=dotted, color=cornflowerblue]'

    output += '\n }'

    g = pydot.graph_from_dot_data(output)
    g = g[0]
    g.write('phenotype'+id+'.svg', prog='dot', format='svg')

def draw_ff(chromosome):
    ''' Draws a feedforward neural network '''

    output = 'digraph G {\n  node [shape=circle, fontsize=9, height=0.2, width=0.2]'

    # subgraph for inputs and outputs
    output += '\n  subgraph cluster_inputs { \n  node [style=filled, shape=box] \n    color=white'
    for ng in chromosome.node_genes:
        if ng.type== 'INPUT':
            output += '\n    '+str(ng.id)
    output += '\n  }'

    output += '\n  subgraph cluster_outputs { \n    node [style=filled, color=lightblue] \n    color=white'
    for ng in chromosome.node_genes:
        if ng.type== 'OUTPUT':
            output += '\n    '+str(ng.id)
    output += '\n  }'
    # topology
    for cg in chromosome.conn_genes:
        output += '\n  '+str(cg.innodeid)+' -> '+str(cg.outnodeid)
        if cg.enabled is False:
            output += ' [style=dotted, color=cornflowerblue]'

    output += '\n }'

    g = pydot.graph_from_dot_data(output)
    g.write('feedforward.svg', prog='dot', format='svg')

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

def plot_spikes(spikes):
    ''' Plots the trains for a single spiking neuron. '''
    time = [i for i in range(len(spikes))]
        
    plot = biggles.FramedPlot()
    plot.title = "Izhikevich's spiking neuron model"
    plot.ylabel = r"Membrane Potential"
    plot.xlabel = r"Time (in ms)"

    plot.add(biggles.Curve(time, spikes, color="green"))
    plot.write_img(600, 300, 'spiking_neuron.svg')
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

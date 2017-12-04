import sys
sys.path.insert(0, '/home/zzhao1/cs81/project-huppili1-zzhao1/experiment/CDNEAT')

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from math import pi, floor
from random import random
from codeepneat import codeepneat, config, population, chromosome, genome, visualize
import pickle
import numpy as np
import keras
from keras.utils import plot_model
from keras.layers import Conv2D, Input
from keras.models import Model

#from .config import Config

def visualize_model(model_file):
    model = keras.models.load_model(model_file)
    layer_list = model.layers
    #module_1 = model.get_layer("model_869")
    #module_2 = model.get_layer("model_870")
    counter = 1
    if len(layer_list) >= 3:
        for module in layer_list[1:-2]:
            plot_model(module, to_file=model_file + '_module_' + str(counter) + '.png', show_shapes = True)
            counter += 1


    #visualize.draw_net(model, "_" + model_file)
    #model.fit(x_train_all, y_train_all, epochs=50)
    plot_model(model, to_file=model_file + '_model.png', show_shapes = True)

def test_nonlinear():

    # input tensor for a 3-channel 256x256 image
    x = Input(shape=(256, 256, 3))
    # 3x3 conv with 3 output channels (same as input channels)
    y = Conv2D(3, (3, 3), padding='same')(x)
    # this returns x + y.
    z = keras.layers.add([x, y])

    model = Model(inputs=x, outputs=z)
    plot_model(model, to_file='nonlinear_model.png', show_shapes = True)




def visualize_module(chromo_file):
    #config.load('configCoverage')
    #chromosome.node_gene_type = genome.NodeGene
    fp = open(chromo_file, "rb")
    #print(fp)
    chromo = pickle.load(fp)
    print(chromo)
    fp.close()
    


    #config.load('configCIFAR10')
    #inputs = keras.layers.Input(config.Config.input_nodes, name='input')
    #module = chromo.decode(inputs)
    #plot_model(module, to_file=chromo_file +'_module.png', show_shapes = True)

#------------------------------------------------------------------
    #visualize.draw_net(chromo, "_" + chromo_file)
    #brain = nn.create_ffphenotype(chromo)
    #fitness = eval_individual(brain, robot, sim, show_trail=True)
    #canvas = Canvas((400,400))
    #sim.physics.draw(canvas)
    #canvas.save("trail_" + chromo_file + ".svg")
    #print("fitness", fitness)

if __name__ == "__main__":
    #directory = "CIFARtest2/"
    # for i in range(11):
    #     model_file = "CIFAR10_best_model_" + str(i)
    #     visualize_model(directory + model_file)
    #     module_file = "CIFAR10_m_best_chromo_" + str(i)
    #     visualize_module(directory + module_file)

    test_nonlinear()

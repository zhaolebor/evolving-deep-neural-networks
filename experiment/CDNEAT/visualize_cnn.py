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
from keras.layers import Conv2D, Input, MaxPooling2D
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

def test_resnet():

    # input tensor for a 3-channel 256x256 image
    x = Input(shape=(256, 256, 3))
    # 3x3 conv with 3 output channels (same as input channels)
    y = Conv2D(3, (3, 3), padding='same')(x)
    # this returns x + y.
    z = keras.layers.add([x, y])

    model = Model(inputs=x, outputs=z)
    plot_model(model, to_file='resnet_model.png', show_shapes = True)

def test_inception():


    input_img = Input(shape=(256, 256, 3))

    tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
    tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
    tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
    tower_3 = Conv2D(64, (1, 1), padding='same', activation='relu')(tower_3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)

    model = Model(inputs=input_img, outputs=output)
    plot_model(model, to_file='inception_model.png', show_shapes = True)




def visualize_module(chromo_file):
    #config.load('configCoverage')
    #chromosome.node_gene_type = genome.NodeGene
    fp = open(chromo_file, "rb")
    #print(fp)
    chromo = pickle.load(fp)
    #print(chromo)
    fp.close()
    print("module:" + chromo_file)
    for connection in chromo._connections:
        print(str(connection))

    config.load('configCIFAR10')
    inputs = keras.layers.Input(config.Config.input_nodes, name='input')
    module = chromo.decode(inputs)
    plot_model(module, to_file=chromo_file +'_module.png', show_shapes = True)

#------------------------------------------------------------------
    #visualize.draw_net(chromo, "_" + chromo_file)
    #brain = nn.create_ffphenotype(chromo)
    #fitness = eval_individual(brain, robot, sim, show_trail=True)
    #canvas = Canvas((400,400))
    #sim.physics.draw(canvas)
    #canvas.save("trail_" + chromo_file + ".svg")
    #print("fitness", fitness)

def mutate_module(chromo_file):
    fp = open(chromo_file, "rb")
    #print(fp)
    chromo = pickle.load(fp)
    #print(chromo)
    fp.close()
    print("module:" + chromo_file)
    for connection in chromo._connections:
        print(str(connection))
    counter = 0
    config.load('configCIFAR10')
    inputs = keras.layers.Input(config.Config.input_nodes, name='input')
    module = chromo.decode(inputs)
    plot_model(module, to_file='mutate/module' + str(counter) + '.png', show_shapes = True)
    while True:
        counter += 1
        mutate = input("Mutate? y/n: ")
        print()
        if mutate == "y":
            chromo._mutate_add_layer()
            for connection in chromo._connections:
                print(str(connection))
            config.load('configCIFAR10')
            inputs = keras.layers.Input(config.Config.input_nodes, name='input')
            module = chromo.decode(inputs)
            plot_model(module, to_file='mutate/module' + str(counter) + '.png', show_shapes = True)
        else:
            return


if __name__ == "__main__":
    directory = "CIFARtest4/"
    for i in range(8):
        model_file = "CIFAR10_best_model_" + str(i)
        visualize_model(directory + model_file)
        module_file = "CIFAR10_m_best_chromo_" + str(i)
        visualize_module(directory + module_file)

    #test_nonlinear()
    #test_inception()

    #module_file = "CIFAR10_m_best_chromo_0"
    #mutate_module(directory + module_file)

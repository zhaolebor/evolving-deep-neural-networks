import sys
sys.path.insert(0, '/home/zzhao1/cs81/project-huppili1-zzhao1/experiment/CDNEAT')

from keras.datasets import cifar10
from math import pi, floor
from random import random
from codeepneat import codeepneat, config, population, chromosome, genome, visualize
import pickle
import numpy as np
import keras
from keras.utils import plot_model
from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras import preprocessing

#from .config import Config
max_words = 10000
(x_train_all, y_train_all), (x_test, y_test) = reuters.load_data(num_words=max_words,
                                                         test_split=0.2)
num_classes = np.max(y_train_all) + 1
x_train = preprocessing.sequence.pad_sequences(x_train_all, maxlen=30)[:8970]
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=30)[:2208]
y_train = keras.utils.to_categorical(y_train_all, num_classes)[:8970]
y_test = keras.utils.to_categorical(y_test, num_classes)[:2208]

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')



def visualize_model(model_file):
    model = keras.models.load_model(model_file)
    layer_list = model.layers
    #module_1 = model.get_layer("model_869")
    #module_2 = model.get_layer("model_870")
    counter = 1
    if len(layer_list) >= 4:
        for module in layer_list[2:-2]:
            plot_model(module, to_file=model_file + '_module_' + str(counter) + '.png', show_shapes = True)
            counter += 1



    #visualize.draw_net(model, "_" + model_file)
    model.fit(x_train, y_train, epochs=20)
    loss, fitness = model.evaluate(x_test, y_test)
    print("fitness", fitness)
    #plot_model(model, to_file=model_file + '_model.png', show_shapes = True)
    #plot_model(module_1, to_file='module_1.png', show_shapes = True)
    #plot_model(module_2, to_file='module_2.png', show_shapes = True)



def visualize_module(chromo_file):
    #config.load('configCoverage')
    #chromosome.node_gene_type = genome.NodeGene
    fp = open(chromo_file, "rb")
    #print(fp)
    chromo = pickle.load(fp)
    print(chromo)
    fp.close()
    config.load('../configCIFAR10')
    inputs = keras.layers.Input(config.Config.input_nodes, name='input')
    module = chromo.decode(inputs)
    plot_model(module, to_file='module.png', show_shapes = True)
    #visualize.draw_net(chromo, "_" + chromo_file)
    #brain = nn.create_ffphenotype(chromo)
    #fitness = eval_individual(brain, robot, sim, show_trail=True)
    #canvas = Canvas((400,400))
    #sim.physics.draw(canvas)
    #canvas.save("trail_" + chromo_file + ".svg")
    #print("fitness", fitness)

if __name__ == "__main__":
    for i in range(10):
        model_file = "run_dec4/reuters_best_model_" + str(i)
        visualize_model(model_file)
    #visualize_model("reuters_best_model_0")

from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from devol import DEvol, GenomeHandler
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import scipy

K.set_image_data_format("channels_last")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train[:,0]
y_test = y_test[:,0]
print("data shapes")
print("  x train:", x_train.shape)
print("  x test:", x_test.shape)
print("  y train:", y_train.shape)
print("  y test:", y_test.shape)
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
dataset = ((x_train, y_train), (x_test, y_test))

max_conv_layers = 6
max_dense_layers = 3 # including final softmax layer
max_conv_kernels = 64
max_dense_nodes = 1024
input_shape = x_train.shape[1:]
num_classes = 10
activ = ["relu"]   # using relu as activation only
#activ = None # using both sigmoid and relu

genome_handler = GenomeHandler(max_conv_layers, max_dense_layers, max_conv_kernels, \
                    max_dense_nodes, input_shape, num_classes, \
                    batch_normalization=True, dropout=True, max_pooling=True, \
                optimizers=None, activations=activ)

num_generations = 50
population_size = 50
num_epochs = 10

devol = DEvol(genome_handler)
model = devol.run(dataset, num_generations, population_size, num_epochs)
model.summary()

import chess_manager as cm
from importlib import reload
import chess
import timeit
import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Dense, Activation,Conv2D,MaxPooling2D,Flatten,Dropout, BatchNormalization
import networks as nn
reload(nn)
reload(cm)

chess_m = cm.ChessManager()
input_batch = chess_m.moves2vec()[0]

# create a batch of 'label' data
#valy = np.random.uniform(low=-1,high=1,size=(input_batch.shape[0]))
valy = np.random.uniform(low = -1, high=1, size=(1,))
poly = np.random.uniform(low=0,high=1,size=(input_batch.shape[0]))

poly = poly / np.sum(poly)

###RUNNING A NETWORK WITH THE SIMPLIFIED INTERFACE

# first you need to get a network and optimizer definition
network = nn.alphago_net((5,8,8), 256, (3,3), 1, (3,3))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

# then create a wrapper object
net = nn.NetworkWrapper(network, optimizer)

# get the predicted pi and z values for a batch of data
out = net.forward(input_batch)

# get the value of the loss function for a batch of data
loss_val = net.forward_loss(input_batch, poly, valy)

# running 100 training steps given a batch
for i in range(100):
    net.training_step(input_batch, poly, valy)

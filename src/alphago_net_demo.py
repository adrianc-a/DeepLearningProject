input_batchimport chess_manager as cm
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

sess = tf.Session()
K.set_session(sess)

g = cm.ChessManager()

# create a 'batch' of the input data
input_batch = g.moves2vec()

# create a batch of 'label' data
valy = np.random.uniform(low=-1,high=1,size=(input_batch.shape[0],1))
poly = np.random.uniform(low=0,high=1,size=(input_batch.shape[0],1))

(inp,valY,polY,pl_out,v_out,loss) = nn.alphago_net((3,8,8), 256, (3,3), 1, 256, (3,3))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

init_op = tf.global_variables_initializer()
sess.run(init_op)

###RUNNING NETWORKS THE MANUAL WAY

# get the predicted output for policy and reward
with sess.as_default():
        # this would be the line to run during generating the tree
        net_out = sess.run((v_out, pl_out), feed_dict={inp:input_batch, K.learning_phase(): 0})

# calculate the actual value of the loss function
with sess.as_default():
        # this gives you the actual loss
        loss_out = sess.run(loss, feed_dict={inp:input_batch, valY:valy,
                                             polY:poly, K.learning_phase(): 0})

k_old = sess.run(tf.trainable_variables()[0])
#notice when I'm running training steps K.learning_phase: 1
with sess.as_default():
	batch = vec
	for i in range(100):
            train_step.run(feed_dict={inp: input_batch, valY:valy, polY:poly, K.learning_phase(): 1})
            k = sess.run(tf.trainable_variables()[0])



###RUNNING A NETWORK WITH THE SIMPLIFIED INTERFACE

# first you need to get a network and optimizer definition
network = nn.alphago_net((3,8,8), 256, (3,3), 1, 256, (3,3))
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

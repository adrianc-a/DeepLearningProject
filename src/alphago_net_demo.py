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

sess = tf.Session()
K.set_session(sess)

# create a 'batch' of the input data
g = cm.ChessManager()
vec = g.state2vec()
new_vec = np.concatenate((vec,vec,vec))

# create a batch of 'label' data 
valy = np.random.uniform(low=-1,high=1,size=(3,1))
poly = np.random.uniform(low=0,high=1,size=(3,1))

(inp,valY,polY,pl_out,v_out,loss) = nn.alphago_net((3,8,8), 256, (3,3), 1, 256, (3,3))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

init_op = tf.global_variables_initializer()
sess.run(init_op)


#note the K.learning_phase: 0 in the feed_dict meaning we're running the net in eval mode
with sess.as_default():
        # this would be the line to run during generating the tree	
        net_out = sess.run((v_out, pl_out), feed_dict={inp:new_vec, K.learning_phase(): 0})
	
        # this gives you the actual loss 
        loss_out = sess.run(loss, feed_dict={inp:new_vec, v_out:valy,
                                             pl_out:poly, K.learning_phase(): 0})


k_old = sess.run(tf.trainable_variables()[0])
#notice when I'm running training steps K.learning_phase: 1
with sess.as_default():
	batch = vec
	for i in range(100):
            train_step.run(feed_dict={inp: new_vec, valY:valy, polY:poly, K.learning_phase(): 1})
            k = sess.run(tf.trainable_variables()[0])


from keras.layers import Dense, Activation, Conv2D, Flatten, BatchNormalization
import keras
import tensorflow as tf
from collections import namedtuple
import keras.backend as K

l2_reg = keras.regularizers.l2

OPTIMIZER_REG = {"adam":tf.train.AdamOptimizer,
                 "sgd":tf.train.GradientDescentOptimizer,
                 "momentum":tf.train.MomentumOptimizer,
                 "rms":tf.train.RMSPropOptimizer}

class NetworkWrapper():

    def __init__(self, network, optimizer):
        """
        Utility class to make interacting with networks easier

        Args:
            network (namedtuple): namedtuple matching the structure of what's
                                  returned by alphago_net, entries should contain
                                  references to input nodes, label nodes, and
                                  the loss function used.

            optimizer (tf.train.Optimizer): a tensorflow optimizer that has
                                            already been initialized with its
                                            parameters
        """
        self.sess = tf.Session()
        K.set_session(self.sess)

        # set all placeholder
        self.input = network.input
        self.policy_label = network.policy_label
        self.value_label = network.value_label

        # get references to policy and value head outputs
        self.value_head = network.value_output
        self.policy_head = network.policy_output

        # get a reference to the computed loss function
        self.loss_function = network.loss

        #self.optimizer = tf.train.AdamOptimizer(lr=learning_rate).minimize(self.loss)
        self.train_step = optimizer.minimize(self.loss_function)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def forward(self, state_batch):
        """
        Given a batch of game states return the output of the policy and value head

        Args:
            state_batch (np.array): batch to run inference on,
                                    1st dimension should be the batch size

        Returns:
            A 2-tuple of 1-d arrays where the first elemenf of the tuple
            corresponds to all the policy head outputs, and the second to all
            the value head outputs for the batch
        """
        with self.sess.as_default():
            net_out = self.sess.run((self.policy_head, self.value_head),
                                     feed_dict={self.input:state_batch, K.learning_phase(): 0})


        return (net_out[0].flatten(), net_out[1].flatten())

    def forward_loss(self, state_batch, policy_batch, value_batch):
        """
        Given batches of states, policy labels (pi), and value labels (z)
        get the loss of the network
        (this corresponds to the (s,pi,z) triples of the paper)

        Args:
            state_batch (np.array): input batch to calculate loss on,
                                    1st dimension should be the batch size
            policy_batch (np.array): policy batch to calculate loss on,
                                    1st dimension should be the batch size
            value_batch (np.array): value batch to calculate loss on,
                                    1st dimension should be the batch size

        Returns:
            A 2-tuple of 1-d arrays where the first elemenf of the tuple
            corresponds to all the value head outputs, and the second all
            the policy head outputs for the batch
        """
        with self.sess.as_default():
            loss_val = self.sess.run(self.loss_function, feed_dict={self.input:state_batch,
                                                 self.value_label:value_batch,
                                                 self.policy_label:policy_batch,
                                                 K.learning_phase(): 0})
        return loss_val

    def training_step(self, state_batch, policy_batch, value_batch):
        """
        Perform a single training step

        Args:
            state_batch (np.array): input batch to calculate loss on,
                                    1st dimension should be the batch size
            policy_batch (np.array): policy batch to calculate loss on,
                                    1st dimension should be the batch size
            value_batch (np.array): value batch to calculate loss on,
                                    1st dimension should be the batch size

        Returns:
        """
        with self.sess.as_default():
            self.train_step.run(feed_dict={self.input:state_batch,
                                      self.value_label:value_batch,
                                      self.policy_label:policy_batch,
                                      K.learning_phase(): 1})




def alphago_net(input_shape, # NOTE: Input shape should be the input size without the resizing for batches
                conv_block_num_filters, conv_block_filter_size,
                num_residual_layers, residual_block_num_filters, residual_block_filter_size,
                policy_head_num_filters=2, policy_head_filter_size=(1,1),
                value_head_num_filters=1, value_head_filter_size=(1,1), regularization=0.001):
    """ Returns a network similar to the one defined in the alphago paper

    conv_block_num_filters: number of filters in the initial conv block
    conv_block_filter_size: filter size in the initial conv block
    num_residual_layers: the number of residual layers to use (in the paper it was 19-40)
    residual_block_num_filters: the number of filters in the residual blocks
    residual_block_filter_size: the size of the filters in the residual block
    regularization: (not required) the l2 regularization strength


    RETURNS: a namedtuple containing all values necessary to run inference, or
    to run training (consult neural_models for an example of how this works
    The elements of the returned namedtuple are as follows:

    input: placeholder op, n the triple (s,pi,z) this would be s
    policy_label: placeholder op, in the triple (s,pi,z) this would be pi
    value_label: placeholder op, in the triple (s,pi,z) this would be z
    policy_output: the output op holding the predicted policy value
    value_output: the output op holding the predicted value of the state
    loss: the output op computing the join cross-entropy/l2 loss of the
        policy and value heads, this is what should be passed into a minimizer
    """


    inp_placeholder_shape = (None,) + input_shape

    valY = tf.placeholder(tf.float32, name='input')
    polY = tf.placeholder(tf.float32, name='policy_label')
    inp = tf.placeholder(tf.float32, shape=inp_placeholder_shape, name='value_label')

    inter_out = convolutional_block(inp,
                                   conv_block_num_filters,
                                   conv_block_filter_size,
                                   input_shape=input_shape,
                                   reg=regularization)


    for i in range(num_residual_layers):
        inter_out = residual_block(inter_out,
                                   residual_block_num_filters,
                                   residual_block_filter_size,
                                   reg=regularization)


    #TODO: possibly allow params here to be changed
    pol_out = policy_head(inter_out, reg=regularization)
    val_out = value_head(inter_out, reg=regularization)

    loss = alphago_loss(pol_out, polY, val_out, valY)

    return namedtuple('Network', 'input policy_label value_label policy_output value_output loss')(*(inp,polY,valY,pol_out,val_out,loss))



def alphago_loss(network_policy, true_policy, network_reward, true_reward):
    return tf.reduce_mean(-true_policy * tf.log(network_policy) + tf.pow(network_reward - true_reward, 2), 0)

def convolutional_block(inp, num_filters, filter_size, input_shape, reg=0.001):

    inp = tf.identity(inp)

    l1 = Conv2D(256, (3,3), padding='same',
                bias_regularizer=l2_reg(reg),
                kernel_regularizer=l2_reg(reg),
                input_shape=input_shape,data_format='channels_first')(inp)

    l2 = BatchNormalization(axis=1)(l1)
    l3 = Activation('relu')(l2)

    return l3


def residual_block(inp, num_filters, filter_size, reg=0.001):
    """
    Num filters: scalar
    filter_size: tuple
    reg: scalar
    inp: tf node, the inpout of the residual_block
    """
    new_in = tf.identity(inp)

    nl1 = Conv2D(num_filters, filter_size, padding='same',
                bias_regularizer=l2_reg(reg),
                kernel_regularizer=l2_reg(reg),
                data_format='channels_first')(new_in)

    nl2 = BatchNormalization(axis=1)(nl1)
    nl3 = Activation('relu')(nl2)
    #nl4 = Conv2D(256, (3,3), padding='same',data_format='channels_first')(nl3)
    nl4 = Conv2D(num_filters, filter_size, padding='same',
                bias_regularizer=l2_reg(reg),
                kernel_regularizer=l2_reg(reg),
                data_format='channels_first')(nl3)

    nl5 = BatchNormalization(axis=1)(nl4)
    nl6 = Activation('relu')(nl5 + new_in)

    return nl6


def policy_head(inp, num_filters=2, filter_size=(1,1), reg=0.001):

    pl_in = tf.identity(inp)
    pl1 = Conv2D(num_filters, filter_size, padding='same',
                 bias_regularizer=l2_reg(reg),
                 kernel_regularizer=l2_reg(reg),
                 data_format='channels_first')(pl_in)
    pl2 = BatchNormalization(axis=1)(pl1)
    pl3 = Activation('relu')(pl2)
    pl_out = Dense(1, activation='sigmoid',
                   bias_regularizer=l2_reg(reg),
                   kernel_regularizer=l2_reg(reg))( Flatten()(pl3) )

    return pl_out


def value_head(inp, num_filters=1, filter_size=(1,1), reg=0.001):

    v_in = tf.identity(inp)
    v1 = Conv2D(num_filters, filter_size, padding='same',
                bias_regularizer=l2_reg(reg),
                kernel_regularizer=l2_reg(reg),
                data_format='channels_first')(v_in)
    v2 = Activation('relu')(v1)
    v_out =  Dense(1,activation='tanh',
                   bias_regularizer=l2_reg(reg),
                   kernel_regularizer=l2_reg(reg))(Flatten()(v2))

    return v_out

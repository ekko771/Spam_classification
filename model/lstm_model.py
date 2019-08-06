import tensorflow as tf
import logging
from utils.setup_logger import logger
logger = logging.getLogger('app.lstm_model')


class Lstm_model(object):
    def __init__(self, config):
        self._n_steps = config['time_step']
        self._input_size = config['num_feature']
        self._output_size = config['num_classes']
        self._layers = config['lstm_layers']
        self._num_units = 256
        self._learning_rate = config['learning_rate']
        self._batch_size = config['batch_size']
        with tf.name_scope('inputs_layer'):
            self.xs = tf.placeholder(
                tf.int32, [None, self._n_steps, self._input_size], name='xs')
            self.ys = tf.placeholder(
                tf.int32, [None, self._output_size], name='ys')
        with tf.variable_scope('embedding_layer'):
            self._embed = self.add_embedding_layer()
        with tf.variable_scope('LSTM_cell'):
            self._rnn_outputs = self.add_cell()
        with tf.variable_scope('out_hidden_layer'):
            self.outputs = self.add_output_layer()
        with tf.name_scope('cost'):
            self.loss = self.compute_lost()
        with tf.name_scope('optimizer'):
            self.train_op = self.optimization()
        logger.info('build model success')

    def add_embedding_layer(self):
        embedding = tf.Variable(tf.random_uniform(
            (10000, 300), -1, 1))
        input_temp = tf.reshape(self.xs, [-1, self._input_size])
        embed = tf.nn.embedding_lookup(embedding, input_temp)
        return embed

    def add_cell(self):
        if isinstance(self._layers[0], dict):
            layers = [tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(layer['num_units'], forget_bias=1.0,
                                        initializer=tf.random_uniform_initializer(
                                        -0.1, 0.1, seed=2), state_is_tuple=True
                                        ),
                layer['keep_prob']
            ) if layer.get('keep_prob') else tf.contrib.rnn.LSTMCell(
                layer['num_units'], forget_bias=1.0,
                initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2), state_is_tuple=True,
            ) for layer in self._layers
            ]
            self._num_units = self._layers[-1]['num_units']
        else:
            return [tf.contrib.rnn.LSTMCell(self._num_units, forget_bias=1.0,
                                            initializer=tf.random_uniform_initializer(
                                                -0.1, 0.1, seed=2),
                                            state_is_tuple=True)
                    for _ in layers]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
        rnn_outputs, states = tf.nn.dynamic_rnn(
            multi_layer_cell, self._embed, dtype=tf.float32)
        return rnn_outputs

    def add_output_layer(self):
        outputs = tf.layers.dense(
            self._rnn_outputs[:, -1], self._output_size, activation=tf.nn.relu)
        return outputs

    def compute_lost(self):
        lost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.outputs, labels=tf.cast(self.ys, tf.float32)))
        return lost

    def optimization(self):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self._learning_rate).minimize(self.loss)
        return optimizer

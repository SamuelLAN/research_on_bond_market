import tensorflow as tf
from lib.tf_models.transformer_base import positional_encoding

import numpy as np

keras = tf.keras
layers = keras.layers


class FC(keras.Model):

    def __init__(self, input_dim, emb_dim, unit_list, dropout_rate=0.0, activation='tanh',
                 mode='concat', use_embeddings=True, name='fc'):
        super(FC, self).__init__(name=name)

        self.__mode = mode
        self.__use_embeddings = use_embeddings
        if self.__use_embeddings:
            # initialize embedding layers
            self.__ranges = np.expand_dims(np.arange(input_dim), axis=0)
            self.__emb = layers.Embedding(input_dim, emb_dim)

            self.d_model = emb_dim
            self.pos_encoding = positional_encoding(20, emb_dim)

        len_layers = len(unit_list)

        # initialize dropout layers
        self.__dropout_layers = [layers.Dropout(dropout_rate, seed=i) for i in range(len_layers)]

        # initialize dense layers
        self.__dense_layers = [layers.Dense(units, activation=activation) for i, units in enumerate(unit_list)]

    def __get_emb(self, x_mask):
        x_mask = tf.expand_dims(tf.cast(x_mask, tf.float32), axis=-1)
        embeddings = self.__emb(self.__ranges) * x_mask
        embeddings = tf.reduce_sum(embeddings, axis=-2)
        return embeddings

    def call(self, inputs, training=None, mask=None):
        if not self.__use_embeddings:
            embeddings = inputs
            embeddings = tf.cast(embeddings, tf.float64)
        else:
            embeddings = self.__get_emb(inputs)

            seq_len = tf.shape(inputs)[1]
            embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
            x_mask = tf.expand_dims(tf.cast(tf.math.greater(tf.reduce_sum(inputs, axis=-1), 0), tf.float32), axis=-1)
            embeddings += self.pos_encoding[:, :seq_len, :] * x_mask

        if self.__mode == 'sum':
            # sum embeddings by their time steps
            x = tf.reduce_sum(embeddings, axis=1)
        else:
            # concat embeddings by their time steps
            x = tf.reshape(embeddings, [-1, embeddings.shape[1] * embeddings.shape[2]])

        for i, dense in enumerate(self.__dense_layers):
            x = self.__dropout_layers[i](x, training=training)
            x = dense(x)

        return x

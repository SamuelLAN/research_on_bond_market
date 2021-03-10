import tensorflow as tf
import numpy as np

keras = tf.keras
layers = keras.layers


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(tf.reduce_sum(seq, axis=-1), 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to th
    # e scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = layers.Dense(d_model, activation='tanh')
        self.wk = layers.Dense(d_model, activation='tanh')
        self.wv = layers.Dense(d_model, activation='tanh')

        self.dense = layers.Dense(d_model, activation='tanh')

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, training):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


class point_wise_feed_forward_network(keras.Model):
    def __init__(self, d_model, d_ff, dropout=0.1, activation='tanh'):
        super(point_wise_feed_forward_network, self).__init__()
        self.dense_1 = layers.Dense(d_ff, activation=activation)
        self.dropout_1 = layers.Dropout(dropout)
        self.dense_2 = layers.Dense(d_model, activation=activation)

    def call(self, inputs, training=None, mask=None):
        x = self.dense_1(inputs)
        x = self.dropout_1(x, training=training)
        x = self.dense_2(x)
        return x


def get_emb(x_mask, emb_layer, ranges):
    x_mask = tf.expand_dims(tf.cast(x_mask, tf.float32), axis=-1)
    embeddings = emb_layer(ranges) * x_mask
    embeddings = tf.reduce_sum(embeddings, axis=-2)
    return embeddings


class EncoderBase(layers.Layer):
    def __init__(self, layer_fn, layer_params, num_layers, d_model, input_vocab_size,
                 maximum_position_encoding, drop_rate=0.1, use_embeddings=True):
        super(EncoderBase, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.__use_embeddings = use_embeddings
        if self.__use_embeddings:
            self.__ranges = np.expand_dims(np.arange(input_vocab_size), axis=0)
            self.embedding = layers.Embedding(input_vocab_size, d_model)
            self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        else:
            self.pos_encoding = positional_encoding(maximum_position_encoding, input_vocab_size)

        self.enc_layers = [layer_fn(*layer_params) for _ in range(num_layers)]

        self.dropout = layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        if self.__use_embeddings:
            embeddings = get_emb(x, self.embedding, self.__ranges)  # (batch_size, input_seq_len, d_model)
            embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        else:
            embeddings = tf.cast(x, tf.float32)

        # x_mask = tf.cast(tf.math.greater(x, 0), tf.float32)
        x_mask = tf.expand_dims(tf.cast(tf.math.greater(tf.reduce_sum(x, axis=-1), 0), tf.float32), axis=-1)
        x = embeddings + self.pos_encoding[:, :seq_len, :] * x_mask

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        # (batch_size, input_seq_len, d_model)
        return x


class DecoderBase(layers.Layer):
    def __init__(self, layer_fn, layer_params, num_layers, d_model, target_vocab_size,
                 maximum_position_encoding, drop_rate=0.1, use_embeddings=True, emb_layer=None):
        super(DecoderBase, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.__use_embeddings = use_embeddings
        if self.__use_embeddings:
            self.__ranges = np.expand_dims(np.arange(target_vocab_size), axis=0)
            if not isinstance(emb_layer, type(None)):
                self.embedding = emb_layer
            else:
                self.embedding = layers.Embedding(target_vocab_size, d_model)
            self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        else:
            self.pos_encoding = positional_encoding(maximum_position_encoding, target_vocab_size)

        self.dec_layers = [layer_fn(*layer_params) for _ in range(num_layers)]
        self.dropout = layers.Dropout(drop_rate)

    def call(self, x, enc_output, end_pos, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        if self.__use_embeddings:
            x = get_emb(x, self.embedding, self.__ranges)  # (batch_size, target_seq_len, d_model)
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        else:
            x = tf.cast(x, tf.float32)

        time_steps = enc_output.shape[1]
        end_pos = tf.squeeze(tf.expand_dims(tf.one_hot(end_pos, time_steps), axis=-1), axis=1)
        pos_embeddings = tf.expand_dims(tf.reduce_sum(self.pos_encoding[:, :time_steps, :] * end_pos, axis=1), axis=1)
        x += pos_embeddings

        # x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class TransformerBase(keras.Model):
    def __init__(self, encoder, decoder, target_vocab_size):
        super(TransformerBase, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.final_layer = layers.Dense(target_vocab_size, activation='tanh')

    def __create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(tar)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask

    def call(self, inputs, training=None, mask=None):
        inp, tar, end_pos = inputs
        # tar = tf.reshape(tar, [tar.shape[0], 1, tar.shape[-1]])
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.__create_masks(inp, tar)

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, end_pos, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        final_output = tf.squeeze(final_output, axis=1)
        # return final_output, attention_weights
        return final_output

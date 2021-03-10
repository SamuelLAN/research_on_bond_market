import tensorflow as tf
from lib.tf_models.transformer_base import EncoderBase, DecoderBase, TransformerBase, \
    MultiHeadAttention, point_wise_feed_forward_network

keras = tf.keras
layers = keras.layers


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, use_embeddings, input_dim, drop_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, drop_rate)
        if use_embeddings:
            self.ffn = point_wise_feed_forward_network(d_model, d_ff, drop_rate)
        else:
            self.ffn = point_wise_feed_forward_network(input_dim, d_ff, drop_rate)

        self.__alpha_1 = tf.Variable(0, trainable=True, name='alpha_1', dtype=tf.float32)
        self.__alpha_2 = tf.Variable(0, trainable=True, name='alpha_2', dtype=tf.float32)

        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask, training=training)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.__alpha_1 * attn_output + x

        ffn_output = self.ffn(out1, training=training)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.__alpha_2 * ffn_output + out1

        return out2, out1


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, use_embeddings, input_dim, drop_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads, drop_rate)
        self.mha2 = MultiHeadAttention(d_model, num_heads, drop_rate)

        if use_embeddings:
            self.ffn = point_wise_feed_forward_network(d_model, d_ff, drop_rate)
        else:
            self.ffn = point_wise_feed_forward_network(input_dim, d_ff, drop_rate)

        self.__alpha_1 = tf.Variable(0, trainable=True, name='alpha_1', dtype=tf.float32)
        self.__alpha_2 = tf.Variable(0, trainable=True, name='alpha_2', dtype=tf.float32)
        self.__alpha_3 = tf.Variable(0, trainable=True, name='alpha_3', dtype=tf.float32)

        self.dropout1 = layers.Dropout(drop_rate)
        self.dropout2 = layers.Dropout(drop_rate)
        self.dropout3 = layers.Dropout(drop_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.__alpha_1 * attn1 + x

        # (batch_size, target_seq_len, d_model)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.__alpha_2 * attn2 + x

        # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.__alpha_3 * ffn_output + out2

        return out3, attn_weights_block1, attn_weights_block2, out1, out2


class Transformer(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size,
                 target_vocab_size, max_pe_input, max_pe_target, drop_rate=0.1, use_embeddings=True,
                 share_embeddings=True):
        super(Transformer, self).__init__()

        self.encoder = EncoderBase(
            EncoderLayer,
            [d_model, num_heads, d_ff, use_embeddings, input_vocab_size, drop_rate],
            num_layers,
            d_model,
            input_vocab_size,
            max_pe_input,
            drop_rate,
            use_embeddings
        )

        self.decoder = DecoderBase(
            DecoderLayer,
            [d_model, num_heads, d_ff, use_embeddings, target_vocab_size, drop_rate],
            num_layers,
            d_model,
            target_vocab_size,
            max_pe_target,
            drop_rate,
            use_embeddings,
            self.encoder.embedding if share_embeddings else None
        )

        self.transformer = TransformerBase(self.encoder, self.decoder, target_vocab_size)

    def call(self, inputs, training=None, mask=None):
        return self.transformer(inputs, training=training, mask=mask)

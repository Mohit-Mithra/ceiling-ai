import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend
from base_rnn import BaseRNN

class Attention(keras.layers.Layer):
    """Attention layer"""
        
    def __init__(self, dim, use_weight=False, hidden_size=512):
        super(Attention, self).__init__()
        self.use_weight = use_weight
        self.hidden_size = hidden_size
        if use_weight:
            print('| using weighted attention layer')
            self.attn_weight = keras.layers.Dense(hidden_size, input_shape=(hidden_size,), use_bias=False)
        self.linear_out = keras.layers.Dense(dim, input_shape=(2*dim,))

    def call(self, output, context):
        """
        - args
        output : Tensor
            decoder output, dim (batch_size, output_size, hidden_size)
        context : Tensor
            context vector from encoder, dim (batch_size, input_size, hidden_size)
        - returns
        output : Tensor
            attention layer output, dim (batch_size, output_size, hidden_size)
        attn : Tensor
            attention map, dim (batch_size, output_size, input_size)
        """
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        if self.use_weight:
            output1 = tf.reshape(output.contiguous(), [-1, hidden_size])
            output = tf.reshape(self.attn_weight(output1), [batch_size, -1, hidden_size])
            
        attn = tf.linalg.matmul(output, context.transpose(1, 2))
        attn1 = keras.activations.softmax(tf.reshape(attn, [-1, input_size]), axis=1)
        attn = tf.reshape(attn1, [batch_size, -1, input_size]) # (batch_size, output_size, input_size)

        mix = tf.linalg.matmul(attn, context) # (batch_size, output_size, hidden_size)
        comb = tf.concat((mix, output), axis=2) # (batch_size, output_size, 2*hidden_size)
        output1 = self.linear_out(tf.reshape(comb, [-1, 2*hidden_size]))
        output = tf.reshape(keras.activations.tanh(output1), [batch_size, -1, hidden_size]) # (batch_size, output_size, hidden_size)

        return output, attn
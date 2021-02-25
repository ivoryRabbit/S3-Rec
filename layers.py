import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization

class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    https://www.tensorflow.org/tutorials/text/transformer 참조하였습니다.
    '''
    def __init__(self, latent_dim, n_head, dropout_rate, epsilon):
        super(MultiHeadAttention, self).__init__() 
        assert latent_dim % n_head == 0
        
        self.latent_dim = latent_dim
        self.n_head = n_head
        self.head_dim = latent_dim // n_head # latent_dim = n_head * head_dim
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon

    def build(self, input_shape):
        assert input_shape[-1] == self.latent_dim
        self.W_q = Dense(self.latent_dim, activation = 'linear')
        self.W_k = Dense(self.latent_dim, activation = 'linear')
        self.W_v = Dense(self.latent_dim, activation = 'linear')

        self.output_layer = Dense(self.latent_dim)
        self.dropout_layer = Dropout(self.dropout_rate)
        self.normalize_layer = LayerNormalization(-1, self.epsilon)

    def split_head(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_head, self.head_dim))
        return tf.transpose(x, perm = [0, 2, 1, 3]) # (batch_size, n_head, seq_len, head_dim)

    def scaled_dot_product_attention(self, q, k, v, mask):
        similarity = tf.matmul(q, k, transpose_b = True)  # (batch_size, n_head, seq_len, seq_len)

        d_k = tf.cast(self.head_dim, dtype = tf.float32)
        similarity /= tf.math.sqrt(d_k)
        similarity += (mask * -1e9)
        
        attention_weight = tf.nn.softmax(similarity, axis = -1)  # (batch_size, n_head, seq_len, seq_len)
        output = tf.matmul(attention_weight, v)  # (batch_size, n_head, seq_len, latent_dim)
        return output, attention_weight

    def call(self, inputs, mask, return_attention_weight = False, training = False):
        '''
        inputs.shape = (batch_size, seq_len, latent_dim)
        outputs.shape = (batch_size, seq_len, latent_dim)
        '''
        batch_size = tf.shape(inputs)[0]

        q = self.W_q(inputs)  # (batch_size, seq_len, latent_dim)
        k = self.W_k(inputs)  # (batch_size, seq_len, latent_dim)
        v = self.W_v(inputs)  # (batch_size, seq_len, latent_dim)

        q = self.split_head(q, batch_size)  # (batch_size, n_head, seq_len, head_dim)
        k = self.split_head(k, batch_size)  # (batch_size, n_head, seq_len, head_dim)
        v = self.split_head(v, batch_size)  # (batch_size, n_head, seq_len, head_dim)

        scaled_attention, attention_weight = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])  # (batch_size, seq_len, n_head, head_dim)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.latent_dim))  # (batch_size, seq_len, latent_dim)

        outputs = self.output_layer(concat_attention)  # (batch_size, seq_len, latent_dim)
        if training:
            outputs = self.dropout_layer(outputs)
        outputs = self.normalize_layer(outputs + inputs) # residual and normalize

        if return_attention_weight:
            return outputs, attention_weight
        return outputs

class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, n_unit, dropout_rate, epsilon):
        super(FeedForwardNetwork, self).__init__()
        self.n_unit = n_unit
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.hidden_layer = Dense(self.n_unit, activation = 'relu')
        self.output_layer = Dense(input_shape[-1])
        self.dropout_layer = Dropout(self.dropout_rate)
        self.normalize_layer = LayerNormalization(-1, self.epsilon)
    
    def call(self, inputs, training = False):
        '''
        inputs.shape = (batch_size, seq_len, latent_dim)
        outputs.shape = (batch_size, seq_len, latent_dim)
        '''
        h = self.hidden_layer(inputs)
        outputs = self.output_layer(h)
        if training:
            outputs = self.dropout_layer(outputs)
        outputs = self.normalize_layer(inputs + outputs)
        return outputs

class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_layer, n_head, n_ffn_unit, dropout_rate, epsilon):
        super(Encoder, self).__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_ffn_unit = n_ffn_unit
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.layer_list = []

    def build(self, input_shape):
        for _ in range(self.n_layer):
            self.layer_list.append([
                MultiHeadAttention(
                    input_shape[-1], self.n_head,
                    self.dropout_rate, epsilon = self.epsilon
                ),
                FeedForwardNetwork(
                    self.n_ffn_unit, 
                    self.dropout_rate, self.epsilon
                )
            ])

    def call(self, inputs, mask, training = False):
        x = inputs
        for MHA, FFN in self.layer_list:
            x = MHA(x, mask, training = training)
            x = FFN(x, training = training)
        return x

class PositionEncoder(tf.keras.layers.Layer): # learnable position encoding
    def __init__(self):
        super(PositionEncoder, self).__init__()

    def build(self, input_shape):
        self.P = self.add_weight(
            shape = input_shape[1:],
            initializer = 'zeros',
            regularizer = None,
            trainable = True,
            name = 'PositionEmbedding'
        )

    def call(self, inputs):
        '''
        inputs.shape = (batch_size, seq_len, latent_dim)
        outputs.shape = (batch_size, seq_len, latent_dim)
        '''
        return inputs + self.P[None, :]
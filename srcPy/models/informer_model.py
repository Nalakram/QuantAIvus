import tensorflow as tf
import numpy as np

class ProbSparseAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads):
        super(ProbSparseAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_dense = tf.keras.layers.Dense(d_model)
        self.k_dense = tf.keras.layers.Dense(d_model)
        self.v_dense = tf.keras.layers.Dense(d_model)
        self.out_dense = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        q, k, v = inputs
        batch_size = tf.shape(q)[0]
        
        q = self.q_dense(q)
        k = self.k_dense(k)
        v = self.v_dense(v)
        
        q = tf.reshape(q, [batch_size, -1, self.n_heads, self.d_k])
        k = tf.reshape(k, [batch_size, -1, self.n_heads, self.d_k])
        v = tf.reshape(v, [batch_size, -1, self.n_heads, self.d_k])
        
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, [batch_size, -1, self.d_model])
        return self.out_dense(output)

class InformerModel(tf.keras.Model):
    def __init__(self, input_dim, output_dim, seq_len, pred_len, d_model=512, n_heads=8, e_layers=3):
        super(InformerModel, self).__init__()
        self.encoder = [ProbSparseAttention(d_model, n_heads) for _ in range(e_layers)]
        self.dense = tf.keras.layers.Dense(output_dim)
        self.input_embedding = tf.keras.layers.Dense(d_model)
        self.pos_encoding = self._get_pos_encoding(seq_len, d_model)

    def _get_pos_encoding(self, seq_len, d_model):
        pos = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)
        return tf.constant(pe, dtype=tf.float32)

    def call(self, inputs):
        x = self.input_embedding(inputs)
        x += self.pos_encoding[:tf.shape(x)[1], :]
        
        for layer in self.encoder:
            x = layer([x, x, x])
        
        x = self.dense(x[:, -1, :])
        return x

# Example usage
def build_informer(input_dim, output_dim, seq_len, pred_len):
    model = InformerModel(input_dim, output_dim, seq_len, pred_len)
    model.compile(optimizer='adam', loss='mse')
    return model
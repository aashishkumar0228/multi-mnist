import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred, input_length, label_length):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_small_model(img_height, num_classes=11):
    # Inputs to the model
    input_img = layers.Input(shape=(None, img_height, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(32,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(64,(3, 3),activation="relu",kernel_initializer="he_normal",padding="same",name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
#     new_shape = ((img_width // 4), (img_height // 4) * 64)
    new_shape = (-1, x.shape[2]*x.shape[3])
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(num_classes, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
    output = CTCLayer(name="ctc_loss")(labels, x, input_length, label_length)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels, input_length, label_length], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


def build_big_model(img_height, num_classes=11):
    # Inputs to the model
    input_img = layers.Input(shape=(None, img_height, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(32,(3, 3),padding="same",name="Conv1")(input_img)
    x = layers.ELU(name="elu1")(x)
    x = layers.Conv2D(32,(3, 3),padding="same",name="Conv2")(x)
    x = layers.ELU(name="elu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.2, name="dropout1")(x)

    # Second conv block
    x = layers.Conv2D(64,(3, 3),padding="same",name="Conv3")(x)
    x = layers.ELU(name="elu3")(x)
    x = layers.Conv2D(64,(3, 3),padding="same",name="Conv4")(x)
    x = layers.ELU(name="elu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.2, name="dropout2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    # new_shape = ((img_width // 4), (img_height // 4) * 64)
    new_shape = (-1, x.shape[2]*x.shape[3])
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2, name="dropout3")(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25), name="bidirectional_1")(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25), name="bidirectional_2")(x)

    # Output layer
    x = layers.Dense(num_classes, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
    output = CTCLayer(name="ctc_loss")(labels, x, input_length, label_length)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels, input_length, label_length], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model

def build_big_model_no_compile(img_height, num_classes=11):
    # Inputs to the model
    input_img = layers.Input(shape=(None, img_height, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(32,(3, 3),padding="same",name="Conv1")(input_img)
    x = layers.ELU(name="elu1")(x)
    x = layers.Conv2D(32,(3, 3),padding="same",name="Conv2")(x)
    x = layers.ELU(name="elu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.2, name="dropout1")(x)

    # Second conv block
    x = layers.Conv2D(64,(3, 3),padding="same",name="Conv3")(x)
    x = layers.ELU(name="elu3")(x)
    x = layers.Conv2D(64,(3, 3),padding="same",name="Conv4")(x)
    x = layers.ELU(name="elu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.2, name="dropout2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    # new_shape = ((img_width // 4), (img_height // 4) * 64)
    new_shape = (-1, x.shape[2]*x.shape[3])
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2, name="dropout3")(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25), name="bidirectional_1")(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25), name="bidirectional_2")(x)

    # Output layer
    x = layers.Dense(num_classes, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
    output = CTCLayer(name="ctc_loss")(labels, x, input_length, label_length)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels, input_length, label_length], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    # opt = keras.optimizers.Adam()
    # moving_avg_opt = tfa.optimizers.MovingAverage(opt)
    # # Compile the model and return
    # model.compile(optimizer=moving_avg_opt)
    return model


## Transformer Related Functions
# Using custom functions for Multi-Head attention layers as the layers.MultiheadAttention API
# requires special ops to convert to tflite.

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
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


def scaled_dot_product_attention(q, k, v, mask=None):
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

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output

class TransformerBlock(layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, maximum_position_encoding, rate=0.1, name=None):
        super().__init__(name=name)
        self.att = MultiHeadAttention(num_heads=num_heads, d_model=key_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(ff_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.pos_encoding = positional_encoding(maximum_position_encoding, ff_dim)

    def call(self, inputs, training):
        seq_len = tf.shape(inputs)[1]
        inputs += self.pos_encoding[:, :seq_len, :]
        attn_output = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_transformer_ocr_model(img_height, num_classes=11):
    # Inputs to the model
    input_img = layers.Input(shape=(None, img_height, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(32,(3, 3),padding="same",name="Conv1")(input_img)
    x = layers.ELU(name="elu1")(x)
    x = layers.Conv2D(32,(3, 3),padding="same",name="Conv2")(x)
    x = layers.ELU(name="elu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Dropout(0.2, name="dropout1")(x)

    # Second conv block
    x = layers.Conv2D(64,(3, 3),padding="same",name="Conv3")(x)
    x = layers.ELU(name="elu3")(x)
    x = layers.Conv2D(64,(3, 3),padding="same",name="Conv4")(x)
    x = layers.ELU(name="elu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Dropout(0.2, name="dropout2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    # new_shape = ((img_width // 4), (img_height // 4) * 64)
    new_shape = (-1, x.shape[2]*x.shape[3])
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2, name="dropout3")(x)

    # Transformers
    transformer_block_1 = TransformerBlock(num_heads=2, key_dim=128, ff_dim=128, maximum_position_encoding=80, name='tansformer_block_1')
    x = transformer_block_1(x)
    transformer_block_2 = TransformerBlock(num_heads=2, key_dim=128, ff_dim=128, maximum_position_encoding=80, name='tansformer_block_2')
    x = transformer_block_2(x)

    # Output layer
    x = layers.Dense(num_classes, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
    output = CTCLayer(name="ctc_loss")(labels, x, input_length, label_length)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels, input_length, label_length], outputs=output, name="ocr_model_v1"
    )
    return model
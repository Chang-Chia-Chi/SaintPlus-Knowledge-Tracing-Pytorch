import random
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def create_padding_mask(seqs):
  # We mask only those vectors of the sequence in which we have all zeroes 
  # (this is more scalable for some situations).
    mask = tf.cast(tf.reduce_all(tf.math.equal(seqs, 0), axis=-1), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def look_head_mask(seq_len):
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype('float32')
    return mask

def pos_encoding(seq_len, model_dim):
    """
    Position encoding function to convert sequence order to model_dim encoding
    based on fomula as below:
    
    PE(pos, 2i) = sin(pos/10000^(2i/model_dim))
    PE(pos, 2i+1) = cos(pos/10000^(2i/model_dim))
    """
    def compute_angle(pos, i, model_dim):
        pos *= tf.cast(1/np.power(10000, 2*(i//2)/model_dim), tf.float32)
        return pos
        
    positions = np.arange(seq_len)[:, np.newaxis] # [seq_len, 1]
    embed_v = np.arange(model_dim)[np.newaxis, :] # [1, model_dim]
    angles = compute_angle(positions, embed_v, model_dim) # [seq_len, model_dim]
    
    pos_enc = np.zeros((seq_len, model_dim))
    pos_enc[:, 0::2] = np.sin(angles[:, 0::2])
    pos_enc[:, 1::2] = np.cos(angles[:, 1::2])
    pos_enc = pos_enc[np.newaxis, ...] # [1, seq_len, model_dim]
    return pos_enc

def scaled_dot_product_attention(q, k, v, pad_mask, head_mask):
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
    if pad_mask is not None:
        scaled_attention_logits += (pad_mask * -1e9)  
        
    if head_mask is not None:
        scaled_attention_logits += (head_mask * -1e9) 
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def plot_result(train_loss, train_auc, val_loss, val_auc):
    fig = plt.figure(figsize=(12, 18))
    epochs = [i for i in range(len(train_loss))]
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.legend(loc="upper right")
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_auc, label="train_aucs")
    plt.plot(epochs, val_auc, label="val_aucs")
    plt.legend(loc="upper right")
    plt.savefig("result.jpg")
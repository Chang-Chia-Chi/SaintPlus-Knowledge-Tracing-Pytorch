import random
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

def create_padding_mask(seqs):
    """
    create zero padding mask
    
    input dimension: [batch_size, seq_len]
    output dimension: [batch_size, 1, 1, seq_len]
    (expand axis for scaled dot attention broadcast [batch, n_heads, seq_q, seq_k])

    ex:
    test_arr1 = np.array([0]*6+[1]*4)
    test_arr2 = np.array([0]*3+[1]*7)
    test_arr = np.stack((test_arr1, test_arr2), axis=0) #(2, 10) (batch, seq)

    create_padding_mask(test_arr):
    tf.Tensor(
    [[[[1. 1. 1. 1. 1. 1. 0. 0. 0. 0.]]]
    [[[1. 1. 1. 0. 0. 0. 0. 0. 0. 0.]]]], shape=(2, 1, 1, 10), dtype=float32)

    """
    mask = tf.cast(tf.math.equal(seqs, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def look_head_mask(seq_len, current_excluded=False):
    """
    create mask to prevent encoder and decoder from recieving information from future data

    params:
    seq_len: sequence length of key
    """
    k = 1 if current_excluded else 0
    mask = np.triu(np.ones((seq_len, seq_len)), k=k).astype('float32')
    return mask

def seq_mask(seqs, current_excluded=False):
    """
    create final mask combining both zero_padding and look_head mask

    params:
    seqs --> seqence to be used for computing mask

    Return:
    sequence mask [batch, 1, seq_q, seq_k]
    """
    seq_len = seqs.shape[1]
    zero_mask = create_padding_mask(seqs)
    head_mask = look_head_mask(seq_len, current_excluded)

    seq_mask = tf.maximum(zero_mask, head_mask)
    return seq_mask

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
    output [batch, num_heads, seq_q, depth]
    attention_weights [batch, num_heads, seq_q, seq_k]
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

def get_time_lag(df):
    """
    Compute time_lag feature, same task_container_id shared same timestamp for each user
    """
    time_dict = {}
    time_lag = np.zeros(len(df), dtype=np.float32)
    for idx, row in enumerate(df[["user_id", "timestamp", "task_container_id"]].values):
        if row[0] not in time_dict:
            time_lag[idx] = 0
            time_dict[row[0]] = [row[1], row[2], 0] # last_timestamp, last_task_container_id, last_lagtime
        else:
            if row[2] == time_dict[row[0]][1]:
                time_lag[idx] = time_dict[row[0]][2]
            else:
                time_lag[idx] = row[1] - time_dict[row[0]][0]
                time_dict[row[0]][0] = row[1]
                time_dict[row[0]][1] = row[2]
                time_dict[row[0]][2] = time_lag[idx]

    df["time_lag"] = time_lag/1000/60 # convert to miniute
    df["time_lag"] = df["time_lag"].clip(0, 1440) # clip to 1440 miniute which is one day
    return time_dict

def compute_loss(loss_fn, labels, preds):
    mask = labels != -1
    labels = labels[mask]
    preds = preds[mask]
    loss = loss_fn(labels, preds)

    return loss
    
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
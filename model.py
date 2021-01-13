import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from data_generator import data_generator
from utils import create_padding_mask, look_head_mask, pos_encoding, scaled_dot_product_attention, seq_mask

"""
Reference:

https://arxiv.org/abs/2002.07033
https://www.tensorflow.org/tutorials/text/transformer
https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/210171
https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html
"""

class MultiHeadAttention(tf.keras.layers.Layer):
    """
    將 Q、K 以及 V 這三個張量先個別轉換到 d_model 維空間，再將其拆成多個比較低維的 depth 維度 N 次以後，
    將這些產生的小 q、小 k 以及小 v 分別丟入前面的注意函式得到 N 個結果。接著將這 N 個 heads 的結果串接起來，
    最後通過一個線性轉換就能得到 multi-head attention 的輸出
    """
    def __init__(self, d_model, num_heads):
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

    def call(self, v, k, q, mask):
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

        return output, attention_weights

class PosEncoding(keras.layers.Layer):
    def __init__(self, d_model, max_len, dropout):
        super(PosEncoding, self).__init__()
        self.dropout = keras.layers.Dropout(dropout)
        self.scale = tf.Variable(1.)

        pe = np.zeros([max_len, d_model], dtype=np.float32)
        pos = np.expand_dims(np.arange(0, max_len, dtype=np.float32), axis=1)
        angle = np.exp(np.arange(0, d_model, 2, dtype=np.float32)*(-np.log(10000.)/d_model))
        pe[:, 0::2] = np.sin(pos*angle)
        pe[:, 1::2] = np.cos(pos*angle)
        pe = np.expand_dims(pe, axis=0)
        self.pe = tf.Variable(pe, trainable=False)

    def call(self, x, training):
        x = x + self.scale*self.pe[:, :x.shape[1], :]
        return self.dropout(x, training=training)

class FFN(keras.layers.Layer):
    def __init__(self, d_ffn, d_model):
        super(FFN, self).__init__()
        self.dense_1 = keras.layers.Dense(d_ffn) #[batch, seq_len, ffn_dim]
        self.dense_2 = keras.layers.Dense(d_model) #[batch, seq_len, d_model]
    
    def call(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x

class ConcatLayer(keras.layers.Layer):
    def __init__(self, d_model, dropout):
        super(ConcatLayer, self).__init__()
        self.dense = keras.layers.Dense(d_model)
        self.dropout = keras.layers.Dropout(dropout)
    def call(self, x):
        x = self.dense(x)
        return self.dropout(x)

class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_ffn, d_model, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(d_ffn, d_model)

        self.dropout_1 = keras.layers.Dropout(dropout)
        self.dropout_2 = keras.layers.Dropout(dropout)

        self.layernorm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = keras.layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, x, mask, training):
        mha_output, _ = self.mha(x, x, x, mask)
        mha_output = self.dropout_1(mha_output, training=training)

        ffn_input = self.layernorm_1(x + mha_output)
        ffn_output = self.ffn(ffn_input)
        ffn_output = self.dropout_2(ffn_output, training=training)

        final_output = self.layernorm_2(ffn_input+ffn_output)
        return final_output

class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_ffn, d_model, num_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.mha_1 = MultiHeadAttention(d_model, num_heads)
        self.mha_2 = MultiHeadAttention(d_model, num_heads)

        self.dropout_1 = keras.layers.Dropout(dropout)
        self.dropout_2 = keras.layers.Dropout(dropout)
        self.dropout_3 = keras.layers.Dropout(dropout)

        self.layernorm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_3 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = FFN(d_ffn, d_model)
    
    def call(self, x, enc_out, mask_1, mask_2, training):
        mha1_output, _ = self.mha_1(x, x, x, mask_1)
        mha1_output = self.dropout_1(mha1_output, training=training)

        mha2_input = self.layernorm_1(x + mha1_output)
        mha2_output, _ = self.mha_2(enc_out, enc_out, mha2_input, mask_2)
        mha2_output = self.dropout_2(mha2_output, training=training)

        ffn_input = self.layernorm_2(mha2_input + mha2_output)
        ffn_output = self.ffn(ffn_input)
        ffn_output = self.dropout_3(ffn_output, training=training)

        final_output = self.layernorm_3(ffn_input + ffn_output)
        return final_output

class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, d_ffn, d_model, num_heads, max_len, n_questions, n_parts, n_tasks, n_ans, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.concat = ConcatLayer(d_model, dropout)
        self.pos_enc = PosEncoding(d_model, max_len, dropout)
        self.dropout = keras.layers.Dropout(dropout)
        self.attention = [EncoderLayer(d_ffn, d_model, num_heads, dropout) for _ in range(num_layers)]
        
        # content related feature
        self.contentId_emb = keras.layers.Embedding(n_questions+1, d_model)
        self.part_emb = keras.layers.Embedding(n_parts+1, d_model)
        self.task_emb = keras.layers.Embedding(n_tasks+1, d_model)
        # time related feature
        self.timelag_emb = keras.layers.Dense(d_model, use_bias=False)
        self.elapsedT_emb = keras.layers.Dense(d_model, use_bias=False)
        # answer related feature
        self.answerCorr_emb = keras.layers.Embedding(3, d_model)
        self.explan_emb = keras.layers.Embedding(3, d_model)
        self.answer_emb = keras.layers.Embedding(n_ans+1, d_model)
    
    def call(self, x, mask, training):
        seq_len = x["e_content_id"].shape[1]

        content_id = self.contentId_emb(x["e_content_id"])
        part = self.part_emb(x["e_part"])
        task = self.task_emb(x["e_task_container_id"])

        time_lag = np.log(x["e_time_lag"]+1)
        time_lag = np.reshape(time_lag, (-1, 1)) # [batch*seq_len, 1]
        time_lag = self.timelag_emb(time_lag) # [batch*seq_len, d_model]
        time_lag = tf.reshape(time_lag, (-1, seq_len, self.d_model)) # [batch, seq_len, d_model]
        elapsed_time = np.log(x["elapsed_time"]+1)
        elapsed_time = np.reshape(elapsed_time, (-1, 1)) # [batch*seq_len, 1]
        elapsed_time = self.elapsedT_emb(elapsed_time) # [batch*seq_len, d_model]
        elapsed_time = tf.reshape(elapsed_time, (-1, seq_len, self.d_model)) # [batch, seq_len, d_model]

        answer_correct = self.answerCorr_emb(x["answer_correct"])
        explain = self.explan_emb(x["explaination"])
        answer = self.answer_emb(x["answer"])

        concat_inp = tf.concat([content_id, part, task, time_lag, elapsed_time, answer_correct, explain, answer], axis=-1)
        concat_out = self.concat(concat_inp)
        concat_out = self.pos_enc(concat_out)

        outputs = self.dropout(concat_out)
        for i in range(self.num_layers):
            outputs = self.attention[i](outputs, mask, training)

        return outputs

class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, d_ffn, d_model, num_heads, max_len, n_questions, n_parts, n_tasks, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.concat = ConcatLayer(d_model, dropout)
        self.pos_enc = PosEncoding(d_model, max_len, dropout)
        self.dropout = keras.layers.Dropout(dropout)
        self.attention = [DecoderLayer(d_ffn, d_model, num_heads, dropout) for _ in range(num_layers)]

        # content related feature
        self.contentId_emb = keras.layers.Embedding(n_questions+1, d_model)
        self.part_emb = keras.layers.Embedding(n_parts+1, d_model)
        self.task_emb = keras.layers.Embedding(n_tasks+1, d_model)
        # time related feature
        self.timelag_emb = keras.layers.Dense(d_model, use_bias=False)
    
    def call(self, x, enc_out, mask_1, mask_2, training):
        seq_len = x["d_content_id"].shape[1]

        content_id = self.contentId_emb(x["d_content_id"])
        part = self.part_emb(x["d_part"])
        task = self.task_emb(x["d_task_container_id"])

        time_lag = np.log(x["d_time_lag"]+1)
        time_lag = np.reshape(time_lag, (-1, 1)) # [batch*seq_len, 1]
        time_lag = self.timelag_emb(time_lag) # [batch*seq_len, d_model]
        time_lag = tf.reshape(time_lag, (-1, seq_len, self.d_model)) # [batch, seq_len, d_model]

        concat_inp = tf.concat([content_id, part, task, time_lag], axis=-1)
        concat_out = self.concat(concat_inp)
        concat_out = self.pos_enc(concat_out)

        outputs = self.dropout(concat_out)
        for i in range(self.num_layers):
            outputs = self.attention[i](outputs, enc_out, mask_1, mask_2, training)

        return outputs

class SaintPlus(keras.Model):
    def __init__(self, num_layers, d_ffn, d_model, num_heads, max_len, n_questions, n_parts, n_tasks, n_ans, dropout=0.1):
        super(SaintPlus, self).__init__()
        self.encoder = Encoder(num_layers, d_ffn, d_model, num_heads, max_len, n_questions, n_parts, n_tasks, n_ans, dropout)
        self.decoder = Decoder(num_layers, d_ffn, d_model, num_heads, max_len, n_questions, n_parts, n_tasks, dropout)
        self.final_layer = keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, x, training):
        mask_1 = seq_mask(x["d_content_id"], current_excluded=False)
        mask_2 = seq_mask(x["d_content_id"], current_excluded=True)

        enc_out = self.encoder(x, mask_1, training)
        dec_out = self.decoder(x, enc_out, mask_1, mask_2, training)
        final_out = self.final_layer(dec_out)

        return final_out

class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

if __name__=='__main__':
    import random
    test_arr1 = np.array([0]*6+[random.randint(1, 9) for _ in range(4)])

    mask_1 = seq_mask(test_arr1[np.newaxis, :], current_excluded=False)
    mask_2 = seq_mask(test_arr1[np.newaxis, :], current_excluded=True)
    
    array = np.array([[0]*6+[2]*4, [0]*5+[2]*5])
    embed = keras.layers.Embedding(10, 4)
    x = {
        "e_content_id": array[:,:-1],
        "e_part": array[:,:-1],
        "e_task_container_id": array[:,:-1],
        "e_time_lag": array[:,:-1],
        "d_content_id": array[:,1:],
        "d_part": array[:,1:],
        "d_task_container_id": array[:,1:],
        "d_time_lag": array[:,1:],
        "elapsed_time": array[:,:-1],
        "answer_correct":array[:,:-1],
        "explaination":array[:,:-1],
        "answer":array[:,:-1]
    }
    
    saint = SaintPlus(4, 1024, 256, 4, 100, 100, 7, 100, 4)
    preds = saint(x)
    saint.summary()
    preds = tf.squeeze(preds, axis=-1)
    loss_fn = keras.losses.BinaryCrossentropy(reduction='none')
    labels = tf.constant([[-1, -1, -1, -1, -1, 0, 1, 1, 1], [-1, -1, -1, -1, 0, 0, 1, 1, 1]])
    loss_mask = labels != -1
    preds = preds[loss_mask]
    labels = labels[loss_mask]
    loss = loss_fn(labels, preds)
    print(labels)
    print(preds)

    
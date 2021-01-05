import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from data_generator import data_generator
from utils import create_padding_mask, look_head_mask, pos_encoding, scaled_dot_product_attention

"""
Reference:

https://www.tensorflow.org/tutorials/text/transformer
https://arxiv.org/abs/2002.07033
"""
class FFN(keras.layers.Layer):
    def __init__(self, model_dim, ffn_out_dim):
        super(FFN, self).__init__()
        self.linear_1 = keras.layers.Dense(ffn_out_dim, activation='relu')
        self.linear_2 = keras.layers.Dense(model_dim)
    
    def call(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        return x

class MultiHeadAttention(keras.layers.Layer):
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

    def call(self, v, k, q, pad_mask, head_mask):
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
            q, k, v, pad_mask, head_mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output

class MultiHeadAttention(keras.layers.Layer):
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

    def call(self, v, k, q, pad_mask, head_mask):
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
            q, k, v, pad_mask, head_mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output

class EncoderLayer(keras.layers.Layer):
    def __init__(self, num_heads, model_dim, ffn_out_dim, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.layernorm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.multi_head = MultiHeadAttention(num_heads, model_dim)
        self.dropout_1 = keras.layers.Dropout(dropout)
        self.layernorm_2 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.ffn = FFN(model_dim, ffn_out_dim)
        self.dropout_2 = keras.layers.Dropout(dropout)
        self.layernorm_3 = keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, x, training, pad_mask, head_mask):
        x = self.layernorm_1(x) # [batch, seq_length, embed_dim]
        mh_1 = self.multi_head(x, x, x, pad_mask, head_mask) #[batch, seq_dim, model_dim]
        mh_1 = self.dropout_1(mh_1, training=training)
        mh_1 = self.layernorm_2(x + mh_1) #[batch, seq_dim, model_dim]
         
        ffn = self.ffn(mh_1) #[batch, seq_dim, model_dim]
        ffn = self.dropout_2(ffn, training=training)
        ffn = self.layernorm_3(mh_1 + ffn) #[batch, seq_dim, model_dim]
        return ffn

class DecoderLayer(keras.layers.Layer):
    def __init__(self, num_heads, model_dim, ffn_out_dim, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.layernorm_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.multi_head_1 = MultiHeadAttention(num_heads, model_dim)
        self.dropout_1 = keras.layers.Dropout(dropout)
        self.layernorm_2 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.multi_head_2 = MultiHeadAttention(num_heads, model_dim)
        self.dropout_2 = keras.layers.Dropout(dropout)
        self.layernorm_3 = keras.layers.LayerNormalization(epsilon=1e-6)        
        
        self.ffn = FFN(model_dim, ffn_out_dim)
        self.dropout_3 = keras.layers.Dropout(dropout)
    
    def call(self, x, en_out, training, pad_mask, head_mask):
        x = self.layernorm_1(x) #[batch, seq_dim, embed_dim]
        mh_1 = self.multi_head_1(x, x, x, pad_mask, head_mask) #[batch, seq_dim, model_dim]
        mh_1 = self.dropout_1(mh_1, training=training)
        mh_1 = self.layernorm_2(x + mh_1)
        
        mh_2 = self.multi_head_2(en_out, en_out, mh_1, pad_mask, head_mask) #[batch, seq_dim, model_dim]
        mh_2 = self.dropout_2(mh_2, training=training)
        mh_2 = self.layernorm_3(mh_1+mh_2)
        
        ffn = self.ffn(mh_2) #[batch, seq_dim, model_dim]
        ffn = self.dropout_3(ffn, training=training)
        ffn = ffn + mh_2 #[batch, seq_dim, model_dim]
        return ffn

class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, num_heads, model_dim, ffn_out_dim, n_questions, n_parts, seq_len, dropout=0.0):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        # self.pos_embed = keras.layers.Embedding(seq_len, model_dim)
        self.exer_embed = keras.layers.Embedding(n_questions+1, model_dim)
        self.part_embed = keras.layers.Embedding(n_parts+1, model_dim)
        self.attentions = [EncoderLayer(num_heads, model_dim, ffn_out_dim, dropout=dropout) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(dropout)
        
    def call(self, exercise, parts, pad_mask, head_mask, training=False):
        exercise = self.exer_embed(exercise)
        parts = self.part_embed(parts)
        # pos = np.arange(exercise.shape[1])
        # pos = self.pos_embed(pos)
        pos = pos_encoding(exercise.shape[1], exercise.shape[-1])
        x = exercise + parts + pos
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x = self.attentions[i](x, training, pad_mask, head_mask)
        return x

class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, num_heads, model_dim, ffn_out_dim, n_responses, n_lag, seq_len, dropout=0.0):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        # self.pos_embed = keras.layers.Embedding(seq_len, model_dim)
        self.correct_embed = keras.layers.Embedding(n_responses, model_dim)
        self.elap_t_embed = keras.layers.Dense(model_dim, use_bias=False) # continuous embedding
        self.lagtime_embed = keras.layers.Embedding(n_lag+1, model_dim)
        self.attentions = [DecoderLayer(num_heads, model_dim, ffn_out_dim, dropout=dropout) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(dropout)
        
    def call(self, en_out, correct, elap_time, lag_time, pad_mask, head_mask, training=False):
        correct = self.correct_embed(correct)
        elaptime = self.elap_t_embed(elap_time)
        lagtime = self.lagtime_embed(lag_time)
        pos = pos_encoding(en_out.shape[1], en_out.shape[-1])
        # pos = np.arange(en_out.shape[1])
        # pos = self.pos_embed(pos)
        x = correct + elaptime + lagtime + pos
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x = self.attentions[i](x, en_out, training, pad_mask, head_mask)
        return x

class Saint(keras.models.Model):
    def __init__(self, num_layers, num_heads, model_dim, ffn_out_dim, n_questions, n_parts, n_responses, n_lag, seq_len, dropout=0.0):
        super(Saint, self).__init__()
        self.encoder = Encoder(num_layers, num_heads, model_dim, ffn_out_dim, n_questions, n_parts, seq_len, dropout=dropout)
        self.decoder = Decoder(num_layers, num_heads, model_dim, ffn_out_dim, n_responses, n_lag, seq_len, dropout=dropout)
        self.final_layer = keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, x, training=False):
        exercise, parts, elap_time, lag_time, correct = x
        x = np.transpose(x, [1,2,0]) # (batch, seq, features)
        pad_mask = create_padding_mask(x)
        head_mask = look_head_mask(x.shape[1])
        elap_time = elap_time[:,:,np.newaxis]
        en_out = self.encoder(exercise, parts, pad_mask, head_mask, training=training)
        de_out = self.decoder(en_out, correct, elap_time, lag_time, pad_mask, head_mask, training=training)
        out = self.final_layer(de_out)
        return out

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class Riiid:
    def __init__(self, num_layers, num_heads, model_dim, ffn_out_dim, n_questions, n_parts, n_responses, n_lag, seq_len, n_features, dropout=0.0):
        self.model = Saint(num_layers, num_heads, model_dim, ffn_out_dim, n_questions, n_parts, n_responses, n_lag, seq_len, dropout=dropout)

        self.loss_fn = keras.losses.BinaryCrossentropy(reduction='none')
        self.acc_fn = keras.metrics.AUC()
        self.train_loss = keras.metrics.Mean()
        self.train_auc = keras.metrics.Mean()
        self.val_loss = keras.metrics.Mean()
        self.val_auc = keras.metrics.Mean()

        learning_rate = CustomSchedule(model_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                                  epsilon=1e-9)
        
        self.seq_len = seq_len
        self.n_features = n_features

    def compute_loss(self, labels, preds):
        mask = tf.math.logical_not(tf.math.equal(labels, 0))
        labels_ = tf.cast(labels-1, tf.int8)
        labels_ = tf.expand_dims(labels_, axis=2)
        preds_ = tf.expand_dims(preds, axis=2)
        loss_ = self.loss_fn(labels_, preds_)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

    def compute_auc(self, labels, preds):
        mask = tf.math.logical_not(tf.math.equal(labels, 0))
        labels_ = tf.cast(labels-1, tf.int8)
        mask = tf.cast(mask, dtype=labels_.dtype)
        auc = self.acc_fn(labels_, preds, sample_weight=mask)
        return auc
    
    def train_step(self, batch_train, batch_labels):
        with tf.GradientTape() as tape:
            preds = self.model(batch_train, training=True)
            preds = tf.squeeze(preds, axis=-1)
            loss = self.compute_loss(batch_labels, preds)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_auc(self.compute_auc(batch_labels, preds))
        return preds, loss

    def valid_step(self, batch_train, batch_labels):
        preds = self.model(batch_train, training=False)
        preds = tf.squeeze(preds, axis=-1)
        loss = self.compute_loss(batch_labels, preds)
        
        self.val_loss(loss)
        self.val_auc(self.compute_auc(batch_labels, preds))
        return preds
    
    def train(self, groups, user_map, epochs=50, patience=3, batch_size=64, partial=0.8, val_ratio=0.04):
        best_auc = 0
        train_loss, train_auc = [], []
        val_loss, val_auc = [], []
        for e in range(epochs):
            t_start = time.time()
            print("==========Epoch {}==========".format(e+1))
            datagen = data_generator(groups, user_map, self.n_features, self.seq_len, partial=partial, val_ratio=val_ratio)
            val_datas = []
            val_labels = []
            print("==========Start Training==========")
            for data in datagen:
                batch_train, batch_val, train_target_id, val_target_id, train_label, val_label = data
                self.train_step(batch_train, train_label)
                val_datas.append(batch_val)
                val_labels.append(val_label)
                
            print("==========Start Validation==========")
            for batch_val, val_label in zip(val_datas, val_labels):
                self.valid_step(batch_val, val_label)
        
            train_loss.append(self.train_loss.result())
            train_auc.append(self.train_auc.result())
            val_loss.append(self.val_loss.result())
            val_auc.append(self.val_auc.result())
            
            t_end = time.time()
            logger = "Traning loss: {:4f} - Training auc: {:4f} / Valid loss: {:4f} - Valid auc: {:4f} / Time spend: {} min"
            print(logger.format(self.train_loss.result(), self.train_auc.result(), self.val_loss.result(), self.val_auc.result(), int((t_end-t_start)/60)))
            
            self.train_auc.reset_states()
            self.val_auc.reset_states()
            
            # early stop
            curr_auc = val_auc[-1]
            if curr_auc.numpy() > best_auc:
                wait = 0
                best_auc = curr_auc
                keras.models.save_model(self.model, "SaintPP")
                keras.models.save_model(self.model, "SaintPP.h5")
                print("Save Model at Epoch {}".format(e+1))
            else:
                wait += 1
                
            if wait >= patience:
                break
                print("Early Stop at Epoch {}".format(e+1))
            
        return train_loss, train_auc, val_loss, val_auc
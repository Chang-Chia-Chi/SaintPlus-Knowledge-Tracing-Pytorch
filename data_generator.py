import gc
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict, deque

class Riiid_Sequence(keras.utils.Sequence):
    def __init__(self, groups, seq_len, user_map, start=0, end=None):
        self.samples = groups
        self.seq_len = seq_len
        self.user_map = user_map
        self.start = start
        self.end = end if end is not None else 0
        
    def __len__(self):
        return self.end - self.start
    
    def __getitem__(self, idx):
        user_id = self.user_map[idx]
        content_id, part, prior_elapsed_time, lag_time, answered_correctly = self.samples[user_id]
        data_len = len(content_id)
        
        q_ids = np.zeros(self.seq_len, dtype=np.int32)
        part_ids = np.zeros(self.seq_len, dtype=np.int8)
        elapsed_ts = np.zeros(self.seq_len, dtype=np.float32)
        lag_ts = np.zeros(self.seq_len, dtype=np.float32)
        ans = np.zeros(self.seq_len, dtype=np.int8)
        
        if data_len > self.seq_len:
            max_idx = random.choice([n for n in range(self.seq_len, data_len)])
            min_idx = max_idx - self.seq_len
            
            q_ids[:] = content_id[min_idx:max_idx]
            part_ids[:] = part[min_idx:max_idx]
            elapsed_ts[:] = prior_elapsed_time[min_idx:max_idx]
            lag_ts[:] = lag_time[min_idx:max_idx]
            ans[:] = answered_correctly[min_idx:max_idx]
        else:
            q_ids[-data_len:] = content_id
            part_ids[-data_len:] = part
            elapsed_ts[-data_len:] = prior_elapsed_time
            lag_ts[-data_len:] = lag_time
            ans[-data_len:] = answered_correctly        
        
        target_ids = q_ids[1:]
        label = ans[1:]
        
        input_ids = q_ids[1:]
        input_parts = part_ids[1:]

        input_rtime = elapsed_ts[:-1]
        input_ans = ans[:-1]
        input_lag = lag_ts[:-1]
        
        inputs = np.concatenate((input_ids[np.newaxis,:], input_parts[np.newaxis,:], input_rtime[np.newaxis,:], 
                                input_lag[np.newaxis,:], input_ans[np.newaxis,:]), axis=0) # features, seq_len
        return inputs, target_ids, label

def training_data(df_pickle_path, n_questions, n_parts, n_lag):
    print("Start Preparing Training Data")
    train_df = pd.read_pickle(df_pickle_path)
    groups = train_df.groupby("user_id").apply(lambda df:(df["content_id"].values, df["part"].values, 
                                                          df["prior_question_elapsed_time"].values, df["lag_time"].values, 
                                                          df["answered_correctly"].values))

    # 產生 group idx 對應 user_id 的字典
    user_map = {i: user_id for i, user_id in enumerate(groups.index)}

    # 加入 start token
    for group in groups:
        group[0][0], group[1][0], group[2][0], group[3][0], group[4][0] = n_questions, n_parts, -99, n_lag, 3
    print("Complete Preparing Training Data")
    return groups, user_map

def data_generator(groups, user_map, n_features, seq_len, batch_size=64, partial=0.8, val_ratio=0.04):
    seq_len = seq_len-1
    # get length of each user and compute probability accordingly
    array_len = []
    for group in groups:
        array_len.append(len(group[0]))
    array_len = np.clip(np.array(array_len), a_min=1, a_max=500)
    prob = array_len/sum(array_len)
    
    # random choose N=# of user with replacement
    ids = [k for k in user_map.keys()]
    takes_ids =  random.choices(ids, weights=prob, k=int(len(user_map)*partial))
    
    # start generate data
    steps = len(takes_ids)//batch_size
    riiid_seq = Riiid_Sequence(groups, seq_len, user_map)
    for s in range(steps):
        batch_inputs = np.zeros((n_features, batch_size, seq_len-1))
        batch_target_id = np.zeros((batch_size, seq_len-1))
        batch_label = np.zeros((batch_size, seq_len-1))
        for j, idx in enumerate(takes_ids[s*batch_size:(s+1)*batch_size]):
            inputs, target_id, label = riiid_seq[idx]
            batch_inputs[:, j, :] = inputs
            batch_target_id[j, :] = target_id
            batch_label[j,:] = label
        
        # divide data into train and val
        n_val = int(seq_len*val_ratio)
        batch_train, batch_val = batch_inputs[:,:,:seq_len-n_val], batch_inputs[:,:,-n_val:]
        train_target_id, val_target_id = batch_target_id[:,:seq_len-n_val], batch_target_id[:,-n_val:]
        train_label, val_label = batch_label[:, :seq_len-n_val], batch_label[:,-n_val:]
        yield tf.convert_to_tensor(batch_train), tf.convert_to_tensor(batch_val),\
              tf.convert_to_tensor(train_target_id), tf.convert_to_tensor(val_target_id),\
              tf.convert_to_tensor(train_label, dtype=tf.float32), tf.convert_to_tensor(val_label, dtype=tf.float32)
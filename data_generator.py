import gc
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict, deque

class Riiid_Sequence(keras.utils.Sequence):
    def __init__(self, groups, seq_len):
        self.samples = {}
        self.seq_len = seq_len
        self.user_ids = []

        for user_id in groups.index:
            c_id, part, t_c_id, t_lag, q_et, ans_c, q_he, u_ans = groups[user_id]
            if len(c_id) < 2:
                continue

            if len(c_id) > self.seq_len:
                initial = len(c_id) % self.seq_len
                if initial > 2:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (
                        c_id[:initial], part[:initial], t_c_id[:initial], t_lag[:initial], 
                        q_et[:initial], ans_c[:initial], q_he[:initial], u_ans[:initial]
                    )
                chunks = len(c_id)//self.seq_len
                for c in range(chunks):
                    start = initial + c*self.seq_len
                    end = initial + (c+1)*self.seq_len
                    self.user_ids.append(f"{user_id}_{c+1}")
                    self.samples[f"{user_id}_{c+1}"] = (
                        c_id[start:end], part[start:end], t_c_id[start:end], t_lag[start:end], 
                        q_et[start:end], ans_c[start:end], q_he[start:end], u_ans[start:end]
                    )
            else:
                self.user_ids.append(f"{user_id}")
                self.samples[f"{user_id}"] = (c_id, part, t_c_id, t_lag, q_et, ans_c, q_he, u_ans)

    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, index):
        user_id = self.user_ids[index]
        c_id, p, t_c_id, t_lag, q_et, ans_c, q_he, u_ans = self.samples[user_id]
        seq_len = len(c_id)

        content_ids = np.zeros(self.seq_len, dtype=int)
        parts = np.zeros(self.seq_len, dtype=int)
        task_container_ids = np.zeros(self.seq_len, dtype=int)
        time_lag = np.zeros(self.seq_len, dtype=float)
        ques_elapsed_time = np.zeros(self.seq_len, dtype=float)
        answer_correct = np.zeros(self.seq_len, dtype=int)
        ques_had_explian = np.zeros(self.seq_len, dtype=int)
        user_answer = np.zeros(self.seq_len, dtype=int)

        if seq_len == self.seq_len:
            content_ids[:] = c_id
            parts[:] = p
            task_container_ids[:] = t_c_id
            time_lag[:] = t_lag
            ques_elapsed_time[:] = q_et
            answer_correct[:] = ans_c
            ques_had_explian[:] = q_he
            user_answer[:] = u_ans
        else:
            content_ids[-seq_len:] = c_id
            parts[-seq_len:] = p
            task_container_ids[-seq_len:] = t_c_id
            time_lag[-seq_len:] = t_lag
            ques_elapsed_time[-seq_len:] = q_et
            answer_correct[-seq_len:] = ans_c
            ques_had_explian[-seq_len:] = q_he
            user_answer[-seq_len:] = u_ans

        return content_ids, parts, task_container_ids, time_lag, ques_elapsed_time, answer_correct, ques_had_explian, user_answer

def data_generator(groups, seq_len, batch_size=256, shuffle=True):
    riiid_seq = Riiid_Sequence(groups, seq_len)
    data_size = len(riiid_seq.user_ids)
    while True:
        indexs = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indexs)

        for i in range(data_size//batch_size):
            b_content_ids = []
            b_parts = []
            b_task_container_ids = []
            b_time_lag = []
            b_ques_elapsed_time = []
            b_answer_correct = []
            b_ques_had_explian = []
            b_user_answer = []

            for idx in indexs[i*batch_size:(i+1)*batch_size]:
                c_ids, parts, t_c_ids, t_lag, ques_et, ans_c, ques_he, u_ans = riiid_seq[idx]
                b_content_ids.append(c_ids)
                b_parts.append(parts)
                b_task_container_ids.append(t_c_ids)
                b_time_lag.append(t_lag)
                b_ques_elapsed_time.append(ques_et)
                b_answer_correct.append(ans_c)
                b_ques_had_explian.append(ques_he)
                b_user_answer.append(u_ans)

            out_dict = {
                "e_content_id": tf.convert_to_tensor(b_content_ids)[:, :-1],
                "d_content_id": tf.convert_to_tensor(b_content_ids)[:, 1:],
                "e_part": tf.convert_to_tensor(b_parts)[:, :-1],
                "d_part": tf.convert_to_tensor(b_parts)[:, 1:],
                "e_task_container_id": tf.convert_to_tensor(b_task_container_ids)[:, :-1],
                "d_task_container_id": tf.convert_to_tensor(b_task_container_ids)[:, 1:],
                "e_time_lag": tf.convert_to_tensor(b_time_lag)[:, :-1],
                "d_time_lag": tf.convert_to_tensor(b_time_lag)[:, 1:],
                "elapsed_time": tf.convert_to_tensor(b_ques_elapsed_time)[:, :-1],
                "answer_correct": tf.convert_to_tensor(b_answer_correct)[:, :-1],
                "explaination": tf.convert_to_tensor(b_ques_had_explian)[:, :-1],
                "answer": tf.convert_to_tensor(b_user_answer)[:, :-1]
            }
            labels = tf.convert_to_tensor([answer_correct[1:]-1 for answer_correct in b_answer_correct])
            yield out_dict, labels

if __name__=="__main__" :
    with open("val_group.pkl.zip", 'rb') as pick:
        val_group = pickle.load(pick)
    
    data_gen = data_generator(val_group, seq_len=100, batch_size=64, shuffle=True)
    for step, (val_train, val_label) in enumerate(data_gen):
        print(val_label)
        if step == 3:
            break
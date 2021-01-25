import gc
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class Riiid_Sequence(Dataset):
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
        label = np.zeros(self.seq_len, dtype=int)
  
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
           
        content_ids = content_ids[1:]
        parts = parts[1:]
        task_container_ids = task_container_ids[1:]
        time_lag = time_lag[1:]
        ques_elapsed_time = ques_elapsed_time[1:]
        label = answer_correct[1:] - 1
        label = np.clip(label, 0, 1)
        
        answer_correct = answer_correct[:-1]
        ques_had_explian = ques_had_explian[1:]
        user_answer = user_answer[:-1]

        return content_ids, parts, time_lag, ques_elapsed_time, answer_correct, ques_had_explian, user_answer, label
import time
import pickle
import numpy as np
import pandas as pd
from utils import get_time_lag

"""
data is from kaggler: tito's strategy

Because this is a time series competition, training and validation dataset should be split by time.
If we only use last several rows for each user as validation, we'll probably focusing too much on light user.
But timestamp feature in original data only specified elapsed time since the user's first event.
We have no idea what's actual time in real world!

tito use a strategy that it first finds maximum timestamp over all users and choose it as upper bound.
Then for each user's own maximum timestamp, Max_TimeStamp subtracts this timestamp to get a interval that
when user might start his/her first event. Finally, random select a time within this interval to get
a virtual start time and add to timestamp feature for each user. 
Sort it by virtual timestamp we could then eazily split train/validation by time.

Only take 30M rows for training/validation
This program will produce three group files: 
1. training 
2. validation
3. inference

and one dictionary time_dict for inference
"""

data_type ={
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32",
    "prior_question_had_explanation": "int8"
}

def pre_process(train_path, ques_path, row_start=30e6, num_rows=30e6, split_ratio=0.9, seq_len=100):
    print("Start pre-process")
    t_s = time.time()

    Features = ["timestamp", "user_id", "content_id", "content_type_id", "task_container_id", "user_answer", 
                "answered_correctly", "prior_question_elapsed_time", "prior_question_had_explanation", "viretual_time_stamp"]
    train_df = pd.read_pickle(train_path)[Features]
    train_df.index = train_df.index.astype('uint32')

    # shift prior elapsed_time and had_explanation to make current elapsed_time and had_explanation
    train_df = train_df[train_df.content_type_id == 0].reset_index()
    train_df["prior_question_elapsed_time"].fillna(0, inplace=True)
    train_df["prior_question_elapsed_time"] /= 1000 # convert to sec
    train_df["prior_question_elapsed_time"] = train_df["prior_question_elapsed_time"].clip(0, 300)
    train_df["prior_question_had_explanation"].fillna(False, inplace=True)
    train_df["prior_question_had_explanation"] = train_df["prior_question_had_explanation"].astype('int8')
    
    # get time_lag feature
    print("Start compute time_lag")
    time_dict = get_time_lag(train_df)
    with open("time_dict.pkl.zip", 'wb') as pick:
        pickle.dump(time_dict, pick)
    print("Complete compute time_lag")
    print("====================")

    train_df = train_df.sort_values(by=["viretual_time_stamp"])
    train_df.drop("timestamp", axis=1, inplace=True)
    train_df.drop("viretual_time_stamp", axis=1, inplace=True)

    print("Start merge dataframe")
    # merge with question dataframe to get part feature
    ques_df = pd.read_csv(ques_path)[["question_id", "part"]]
    train_df = train_df.merge(ques_df, how='left', left_on='content_id', right_on='question_id')
    train_df.drop(["question_id"], axis=1, inplace=True)
    train_df["part"] = train_df["part"].astype('uint8')
    print(train_df.head(10))
    print("Complete merge dataframe")
    print("====================")

    # plus 1 for cat feature which starts from 0
    train_df["content_id"] += 1
    train_df["task_container_id"] += 1
    train_df["answered_correctly"] += 1
    train_df["prior_question_had_explanation"] += 1
    train_df["user_answer"] += 1

    Train_features = ["user_id", "content_id", "part", "task_container_id", "time_lag", "prior_question_elapsed_time",
                      "answered_correctly", "prior_question_had_explanation", "user_answer", "timestamp"]

    if num_rows == -1:
        num_rows = train_df.shape[0]
    train_df = train_df.iloc[int(row_start):int(row_start+num_rows)]
    val_df = train_df[int(num_rows*split_ratio):]
    train_df = train_df[:int(num_rows*split_ratio)]

    print("Train dataframe shape after process ({}, {})/ Val dataframe shape after process({}, {})".format(train_df.shape[0], train_df.shape[1], val_df.shape[0], val_df.shape[1]))
    print("====================")

    # Check data balance
    num_new_user = val_df[~val_df["user_id"].isin(train_df["user_id"])]["user_id"].nunique()
    num_new_content = val_df[~val_df["content_id"].isin(train_df["content_id"])]["content_id"].nunique()
    train_content_id = train_df["content_id"].nunique()
    train_part = train_df["part"].nunique()
    train_correct = train_df["answered_correctly"].mean()-1
    val_correct = val_df["answered_correctly"].mean()-1
    print("Number of new users {}/ Number of new contents {}".format(num_new_user, num_new_content))
    print("Number of content_id {}/ Number of part {}".format(train_content_id, train_part))
    print("train correctness {:.3f}/val correctness {:.3f}".format(train_correct, val_correct))
    print("====================")

    print("Start train and Val grouping")
    train_group = train_df[Train_features].groupby("user_id").apply(lambda df: (
        df["content_id"].values,
        df["part"].values,
        df["task_container_id"].values,
        df["time_lag"].values,
        df["prior_question_elapsed_time"].values,
        df["answered_correctly"].values,
        df["prior_question_had_explanation"].values,
        df["user_answer"].values,
    ))
    with open("train_group.pkl.zip", 'wb') as pick:
        pickle.dump(train_group, pick)
    del train_group, train_df

    val_group = val_df[Train_features].groupby("user_id").apply(lambda df: (
        df["content_id"].values,
        df["part"].values,
        df["task_container_id"].values,
        df["time_lag"].values,
        df["prior_question_elapsed_time"].values,
        df["answered_correctly"].values,
        df["prior_question_had_explanation"].values,
        df["user_answer"].values,
    ))
    with open("val_group.pkl.zip", 'wb') as pick:
        pickle.dump(val_group, pick)
    print("Complete pre-process, execution time {:.2f} s".format(time.time()-t_s))

if __name__=="__main__":
    train_path = ""
    ques_path = ""
    # be aware that appropriate range of data is required to ensure all questions 
    # are in the training set, or LB score will be much lower than CV score
    # Recommend to user all of the data.
    pre_process(train_path, ques_path, 0, -1, 0.95)
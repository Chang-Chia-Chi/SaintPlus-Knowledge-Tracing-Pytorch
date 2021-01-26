# SaintPlus-Knowledge-Tracing-Transformer
## Introduction
Thanks to Kaggle and a lot of amazing data enthusiasm people sharing their notebooks so I had a chance to learn Transformer and really use it to a real-world task!   
    
Saint+ is a **Transformer** based knowledge-tracing model which takes students' exercise history information to predict future performance. As classical Transformer, it has an Encoder-Decoder structure that Encoder applied self-attention to a stream of exercise embeddings; Decoder applied self-attention to responses embeddings and encoder-decoder attention to encoder output.
## SaintPlus
The basic idea is that we fed current question and stream of past exerices into encoder, it will find which parts of exercises experience should be noticed. Then fed the weighted sum of encoder value vector as key and value to encoder-decoder attention layer.    
    
How students performed in past is used as decoder input. The first layer of decoder will learn relationship between responses, how long they took for a question task and time gap between different task user answered. The output sequence from first decoder layer is forward to second layer as query. The intuitive explanation is right now we have past experince of knowledge (query), how will a student perform (weighted value vector) for a sequence of questions (key, value).     
Besides, Causal Mask as shown below is applied to all of encoder & decoder layers to prevent future data leakage.   
![image](https://github.com/Chang-Chia-Chi/SaintPlus-Knowledge-Tracing-Pytorch/blob/main/pics/Causal_Mask.jpg)
![image](https://github.com/Chang-Chia-Chi/SaintPlus-Knowledge-Tracing-Pytorch/blob/main/pics/time_feature.jpg)
## Structure of model
![image](https://github.com/Chang-Chia-Chi/SaintPlus-Knowledge-Tracing-Pytorch/blob/main/pics/my_model.jpg)
## Modification
1. Add **prior_question_had_explanation** to encoder, provide information whether a user watched answer or explanation of last quecstion.    
2. Add **prior_user_answer** to decoder, provide information of answer stream. Like if a student picked same choice of answer (A,A,A,A for instance) for a sequence of questions, he/she is probably be wrong for next question.   
3. Both time features are scaled by **nature log** to help model converge more quickly and easily.    
4. Use **concatenate** instead of plus to combine information from different embeddings and add position embedding with a learnable weighting factor.    
5. Use **Norm learning rate** as from Transformer paper.
## Model and Training parameters
|Parameter                |Value
| ------------------------|-----------
|Number of Attention Layer|2
|Number of Head           |4
|Embedding Dimension      |128
|Forward Linear Layer Dimension|512
|Dropout|0.1
|Max Sequence Length |100
|Batch Size |512
|Warm Steps |4000
|Split Ratio|0.95
## CV Strategy 
Data is derived from kaggler: tito's strategy [notebook](https://www.kaggle.com/its7171/cv-strategy).
Because this is a time series competition, training and validation dataset should be split by time. If we only use last several rows for each user as validation, we'll probably focusing too much on heavy user. But timestamp feature in original data only specified time the question be finished since the user's first event. We have no idea what's actual time in real world!    
tito used a strategy that it first finds maximum timestamp over all users and uses it as upper bound. Then for each user's own maximum timestamp, Max_TimeStamp subtracts this timestamp to get a interval that when user might start his/her first event. Finally, random select a time within this interval to get
a virtual start time and add to timestamp feature for each user. Sort it by virtual timestamp we could then easily split train/validation by time.
## How to train
1. `git clone https://github.com/Chang-Chia-Chi/SaintPlus-Knowledge-Tracing-Pytorch.git`    
2. cd `SaintPlus-Knowledge-Tracing-Pytorch`   
3. Run the kernel below and download data to your host machine.   
[Training/Validation/Inference group data notebook](https://www.kaggle.com/maxchang0724/get-infer)
(P.S. You could also run `pre_process.py` but you have to concatenate train/validation dataframe from [tito](https://www.kaggle.com/its7171/cv-strategy) first and download it. Besides, question csv from [competition](https://www.kaggle.com/c/riiid-test-answer-prediction) is also required. Then set correct file path in `pre_process.py` and run.
4. Adjust hyperparameter in `parser.py`
5. Run `python train.py`
## Training Result
CV 0.799/Private 0.799 AUC    
[kaggle training notebook](https://www.kaggle.com/maxchang0724/saint-training-cv-0-799-private-0-799)   
[kaggle inference notebook](https://www.kaggle.com/maxchang0724/saint-inference-private-0-799)
## Reference
https://arxiv.org/abs/2010.12042 - SAINT+: Integrating Temporal Features for EdNet Correctness Prediction   
https://arxiv.org/abs/1706.03762 - Attention Is All You Need

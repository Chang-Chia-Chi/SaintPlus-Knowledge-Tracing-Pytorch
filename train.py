import time
import pickle
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from parser import parser
from model import SaintPlus, NoamOpt
from torch.utils.data import DataLoader
from data_generator import Riiid_Sequence
from sklearn.metrics import roc_auc_score

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    args = parser.parse_args()
    num_layers = args.num_layers
    num_heads = args.num_heads
    d_model = args.d_model
    d_ffn = d_model*4
    max_len = args.max_len
    n_questions = args.n_questions
    n_parts = args.n_parts
    n_tasks = args.n_tasks
    n_ans = args.n_ans

    seq_len = args.seq_len
    warmup_steps = args.warmup_steps
    dropout = args.dropout
    epochs = args.epochs
    patience = args.patience
    batch_size = args.batch_size

    with open("train_group.pkl.zip", 'rb') as pick:
        train_group = pickle.load(pick)
    with open("val_group.pkl.zip", 'rb') as pick:
        val_group = pickle.load(pick)

    train_seq = Riiid_Sequence(train_group, seq_len)
    train_size = len(train_seq)
    train_loader = DataLoader(train_seq, batch_size=batch_size, shuffle=True, num_workers=8)
    del train_seq, train_group

    val_seq = Riiid_Sequence(val_group, seq_len)
    val_size = len(val_seq)
    val_loader = DataLoader(val_seq, batch_size=batch_size, shuffle=False, num_workers=8)
    del val_seq, val_group

    loss_fn = nn.BCELoss()
    model = SaintPlus(seq_len=100, num_layers=num_layers, d_ffn=d_ffn, d_model=d_model, num_heads=num_heads,
                    max_len=max_len, n_questions=n_questions, n_parts=n_parts, n_tasks=n_tasks, 
                    n_ans=n_ans, dropout=dropout)
    optimizer = NoamOpt(d_model, 1, 4000 ,optim.Adam(model.parameters(), lr=0))
    model.to(device)
    loss_fn.to(device)

    train_losses = []
    val_losses = []
    val_aucs = []
    best_auc = 0
    for e in range(epochs):
        print("==========Epoch {} Start Training==========".format(e+1))
        model.train()
        t_s = time.time()
        train_loss = []
        for step, data in enumerate(train_loader):
            content_ids = data[0].to(device).long()
            parts = data[1].to(device).long()
            time_lag = data[2].to(device).float()
            ques_elapsed_time = data[3].to(device).float()
            answer_correct = data[4].to(device).long()
            ques_had_explian = data[5].to(device).long()
            user_answer = data[6].to(device).long()
            label = data[7].to(device).float()

            optimizer.optimizer.zero_grad()

            preds = model(content_ids, parts, time_lag, ques_elapsed_time, answer_correct, ques_had_explian, user_answer)
            loss_mask = (answer_correct != 0)
            preds_masked = torch.masked_select(preds, loss_mask)
            label_masked = torch.masked_select(label, loss_mask)
            loss = loss_fn(preds_masked, label_masked)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
                
        train_loss = np.mean(train_loss)
        print("==========Epoch {} Start Validation==========".format(e+1))
        model.eval()
        val_loss = []
        val_labels = []
        val_preds = []
        for step, data in enumerate(val_loader):
            content_ids = data[0].to(device).long()
            parts = data[1].to(device).long()
            time_lag = data[2].to(device).float()
            ques_elapsed_time = data[3].to(device).float()
            answer_correct = data[4].to(device).long()
            ques_had_explian = data[5].to(device).long()
            user_answer = data[6].to(device).long()
            label = data[7].to(device).float()

            preds = model(content_ids, parts, time_lag, ques_elapsed_time, answer_correct, ques_had_explian, user_answer)
            loss_mask = (answer_correct != 0)
            preds_masked = torch.masked_select(preds, loss_mask)
            label_masked = torch.masked_select(label, loss_mask)

            val_loss.append(loss.item())
            val_labels.extend(label_masked.view(-1).data.cpu().numpy())
            val_preds.extend(preds_masked.view(-1).data.cpu().numpy())

        val_loss = np.mean(val_loss)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        if val_auc > best_auc:
            print("Save model at epoch {}".format(e+1))
            torch.save(model.state_dict(), "./saint.pt")
            best_auc = val_auc
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        exec_t = int((time.time() - t_s)/60)
        print("Train Loss {:.4f}/ Val Loss {:.4f}, Val AUC {:.4f} / Exec time {} min".format(train_loss, val_loss, val_auc, exec_t))
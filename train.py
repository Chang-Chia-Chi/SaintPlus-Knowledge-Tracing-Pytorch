import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from model import SaintPlus, CustomSchedule
from tqdm import tqdm 
from parser import parser
from utils import plot_result, compute_loss
from data_generator import data_generator, Riiid_Sequence
from sklearn.metrics import roc_auc_score

if __name__=="__main__":

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

    train_riiid = Riiid_Sequence(train_group, seq_len)
    train_steps = len(train_riiid)//batch_size
    print("Training steps {}".format(train_steps))
    del train_riiid
    train_gen = data_generator(train_group, seq_len=100, batch_size=batch_size, shuffle=True)

    val_riiid = Riiid_Sequence(val_group, seq_len)
    val_steps = len(val_riiid)//batch_size
    print("Validation steps {}".format(val_steps))
    del val_riiid
    val_gen = data_generator(val_group, seq_len=100, batch_size=batch_size, shuffle=False)

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, 
                                         epsilon=1e-8)
    loss_fn = keras.losses.BinaryCrossentropy(reduction='none')
    saint = SaintPlus(num_layers=num_layers, d_ffn=d_ffn, d_model=d_model, num_heads=num_heads,
                    max_len=max_len, n_questions=n_questions, n_parts=n_parts, n_tasks=n_tasks, 
                    n_ans=n_ans, dropout=dropout)

    best_auc = 0
    wait = 0
    train_losses = []
    train_aucs = []
    val_losses = []
    val_aucs = []
    for e in range(epochs):
        print("==========Epoch {} Start Training==========".format(e+1))
        t_s = time.time()
        train_loss = []
        train_labels = []
        train_preds = []
        pbar = tqdm(train_steps)
        for step, (batch_train, batch_label) in enumerate(train_gen):
            with tf.GradientTape() as tape:
                preds = saint(batch_train, training=True)
                preds = tf.squeeze(preds, axis=-1)
                loss = compute_loss(loss_fn, batch_label, preds)
            
            grads = tape.gradient(loss, saint.trainable_variables)
            optimizer.apply_gradients(zip(grads, saint.trainable_variables))

            mask = batch_label != -1
            train_loss.append(loss.numpy())
            train_labels.extend(batch_label[mask].numpy())
            train_preds.extend(preds[mask].numpy())
            pbar.update(1)
            if step == train_steps - 1:
                break
        
        train_loss = np.mean(train_loss)
        train_auc = roc_auc_score(train_labels, train_preds)

        print("==========Epoch {} Start Validation==========".format(e+1))
        val_loss = []
        val_labels = []
        val_preds = []
        pbar = tqdm(val_steps)
        for step, (batch_val, batch_label) in enumerate(val_gen):
            preds = saint(batch_val, training=False)
            preds = tf.squeeze(preds, axis=-1)
            loss = compute_loss(loss_fn, batch_label, preds)

            mask = batch_label != -1
            val_loss.append(loss.numpy())
            val_labels.extend(batch_label[mask].numpy())
            val_preds.extend(preds[mask].numpy())
            pbar.update(1)
            if step == val_steps - 1:
                break
        
        val_loss = np.mean(val_loss)
        val_auc = roc_auc_score(val_labels, val_preds)

        train_losses.append(train_loss)
        train_aucs.append(train_auc)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        exec_t = int((time.time() - t_s)/60)
        print("Train Loss {:.4f}, Train AUC {:.4f} / Val Loss {:.4f}, Val AUC {:.4f} / Exec time {} min".format(train_loss, train_auc, val_loss, val_auc, exec_t))

        if val_auc > best_auc:
            best_auc = val_auc
            wait = 0
            saint.save_weights("saintpp")
        else:
            wait += 1
            if wait >= patience:
                print("Early Stopping at Epoch {}".format(e+1))
                break

    plot_result(train_losses, train_aucs, val_losses, val_aucs)
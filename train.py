import pickle
from model import Riiid
from parser import parser
from utils import plot_result
from data_generator import training_data

args = parser.parse_args()
num_layers = args.num_layers
num_heads = args.num_heads
model_dim = args.model_dim
ffn_out_dim = model_dim*4
n_questions = args.n_questions + 1 # plus padding
n_parts = args.n_parts + 1 # plus padding
n_responses = args.n_responses + 2 # plus padding and <SOS>
n_lag = args.n_lag + 1 # plus <SOS> 
seq_len = args.seq_len + 1
n_features = args.n_feat
dropout = args.dropout
epochs = args.epochs
patience = args.patience
batch_size = args.batch_size
partial = args.partial
val_ratio = args.val_ratio

riiid = Riiid(num_layers=num_layers, num_heads=num_heads, model_dim=model_dim,
              ffn_out_dim=ffn_out_dim, n_questions=n_questions, n_parts=n_parts,
              n_responses=n_responses, n_lag=n_lag, seq_len=seq_len, 
              n_features=n_features, dropout=dropout)

with open("groups.pickle", 'rb') as pick:
    groups = pickle.load(pick)
with open("user_map.pickle", 'rb') as pick:
    user_map = pickle.load(pick)

train_loss, train_auc, val_loss, val_auc = riiid.train(groups, user_map, epochs=epochs, patience=patience, 
                                                       batch_size=batch_size, partial=partial, val_ratio=val_ratio)
plot_result(train_loss, train_auc, val_loss, val_auc)
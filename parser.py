import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', type=int, default=4,
                    help='number of multihead attention layer(default: 4)')
parser.add_argument('--num_heads', type=int, default=4,
                    help='number of head in one multihead attention layer(default: 8)')
parser.add_argument('--model_dim', type=int, default=128,
                    help='dimension of embedding size(default: 256)')
parser.add_argument('--n_questions', type=int, default=13523,
                    help='number of different questions(default: 13523)')
parser.add_argument('--n_parts', type=int, default=7,
                    help='number of different parts(default: 7)')
parser.add_argument('--n_responses', type=int, default=2,
                    help='answer correct of not(default: 2)')
parser.add_argument('--n_lag', type=int, default=153,
                    help='number of lag time categories(default: 153)')
parser.add_argument('--seq_len', type=int, default=100,
                    help='sequence length(default: 100)')
parser.add_argument('--n_feat', type=int, default=5,
                    help='number of features used for training(default: 5)')
parser.add_argument('--n_features', type=int, default=5,
                    help='number of features used for training(default: 5)')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout ratio(default: 0.0)')   
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epochs(default: 50)')
parser.add_argument('--patience', type=int, default=3,
                    help='patience to wait before early stopping(default: 3)')   
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size(default: 64)')
parser.add_argument('--partial', type=float, default=0.8,
                    help='ratio of all users used to be trained(default: 0.8)')   
parser.add_argument('--val_ratio', type=float, default=0.04,
                    help='validation ratio(default: 0.04)')   

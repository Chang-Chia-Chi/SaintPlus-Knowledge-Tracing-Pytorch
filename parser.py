import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of multihead attention layer(default: 4)')
parser.add_argument('--num_heads', type=int, default=8,
                    help='number of head in one multihead attention layer(default: 8)')
parser.add_argument('--d_model', type=int, default=256,
                    help='dimension of embedding size(default: 256)')
parser.add_argument('--max_len', type=int, default=1000,
                    help='maximum index for position encoding(default: 1000)')
parser.add_argument('--n_questions', type=int, default=13523,
                    help='number of different question(default: 13523)')
parser.add_argument('--n_parts', type=int, default=7,
                    help='number of different part(default: 7)')
parser.add_argument('--n_tasks', type=int, default=10000,
                    help='number of different task id(default: 10000)')
parser.add_argument('--n_ans', type=int, default=4,
                    help='number of choice of answer(default: 4)')
parser.add_argument('--seq_len', type=int, default=100,
                    help='sequence length(default: 100)')
parser.add_argument('--warmup_steps', type=int, default=4000,
                    help='warmup_steps for learning rate(default: 4000)')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout ratio(default: 0.1)')   
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs(default: 30)')
parser.add_argument('--patience', type=int, default=3,
                    help='patience to wait before early stopping(default: 3)')   
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size(default: 64)')  

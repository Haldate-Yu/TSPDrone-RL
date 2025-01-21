import numpy as np
import os
import random
import torch
import argparse
from model.nnets import Actor, Critic
from utils.agent import A2CAgent
from utils.env_no_comb import Env, DataGenerator
from utils.utils import get_model
from graph_model.base_model import Net_GCN


def str2bool(v):
    return v.lower() in ('true', '1')


if __name__ == '__main__':
    # load args from main
    parser = argparse.ArgumentParser(description="TSP with Drone")

    # Data generation for Training and Testing
    parser.add_argument('--n_nodes', default=11, type=int, help="Number of nodes")
    parser.add_argument('--R', default=150, type=int, help="Drone battery life in time units")
    parser.add_argument('--v_t', default=1, type=int, help="Speed of truck in m/s")
    parser.add_argument('--v_d', default=2, type=int, help="Speed of drone in m/s")
    parser.add_argument('--max_w', default=2.5, type=float, help="Max weight a drone can carry")
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size for training')
    parser.add_argument('--n_train', default=1000000, type=int, help='# of episodes for training')
    parser.add_argument('--test_size', default=100, type=int, help='# of instances for testing')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--save_path', type=str, default='trained_models/')
    parser.add_argument('--test_interval', default=200, type=int, help='test every test_interval steps')
    parser.add_argument('--save_interval', default=1000, type=int, help='save every save_interval steps')
    parser.add_argument('--log_dir', default='logs', type=str, help='folder for saving prints')
    parser.add_argument('--stdout_print', default=True, type=str2bool, help='print control')

    '''
        Neural Network Structure 
    '''
    # Embedding
    parser.add_argument('--embedding_dim', default=3, type=int, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', default=256, type=int, help='Dimension of hidden layers in Enc/Dec')

    # Decoder: LSTM
    parser.add_argument('--rnn_layers', default=1, type=int, help='Number of LSTM layers in the encoder and decoder')
    parser.add_argument('--forget_bias', default=1.0, type=float, help="Forget bias for BasicLSTMCell.")
    parser.add_argument('--dropout', default=0.1, type=float, help='The dropout prob')

    # Attention
    parser.add_argument('--use_tanh', type=str2bool, default=False, help='use tanh before computing probs in attention')
    parser.add_argument('--mask_logits', type=str2bool, default=True, help='mask unavailble nodes probs')

    # Graph Topology
    parser.add_argument('--create_graph', type=str, default='knn', help='way to generate graph')
    parser.add_argument('--k_value', type=int, help='specific k value for knn method')
    parser.add_argument('--distance_threshold', type=float, help='specific threshold value for knn+ method')
    parser.add_argument('--use_coord_features', default=True, type=str2bool, help='whether to use '
                                                                                  'coordinates as node features or'
                                                                                  ' distance as node features')

    # Graph Model Options
    parser.add_argument('--model_name', type=str, default='gcn',
                        choices=['gcn', 'gat', 'sage', 'lp'],
                        help='Type of model')
    parser.add_argument('--gnn_layers', type=int, default=2, help='model layers for gnn model')
    parser.add_argument('--gnn_hidden_dims', type=int, default=32, help='hidden dims for gnn model')

    '''
        Train & Test Options
    '''
    # Training
    parser.add_argument('--train', default=False, type=str2bool, help="whether to do the training or not")
    parser.add_argument('--actor_net_lr', default=1e-4, type=float, help="Set the learning rate for the actor network")
    parser.add_argument('--critic_net_lr', default=1e-4, type=float,
                        help="Set the learning rate for the critic network")
    parser.add_argument('--random_seed', default=42, type=int, help='Random seed for testing')
    parser.add_argument('--max_grad_norm', default=2.0, type=float, help='Gradient clipping')
    parser.add_argument('--decode_len', default=30, type=int, help='Max number of steps per episode')
    parser.add_argument('--patience', type=int, default=200, help='patience for early stopping')

    # Evaluation
    parser.add_argument('--sampling', default=True, type=str2bool, help="whether to do the batch sampling or not")
    parser.add_argument('--n_samples', default=5, type=int, help='the number of samples for batch sampling')

    args, unknown = parser.parse_known_args()
    args = vars(args)
    args['decode_len'] = max(round(args['n_nodes'] * 1.8), args['decode_len'])

    # print args
    # for key, value in sorted(args.items()):
    #     print("{}: {}".format(key, value))

    # args = ParseParams()
    # seed everything
    random_seed = args['random_seed']
    if random_seed is not None and random_seed > 0:
        print("# Set random seed to %d" % random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # loading basic args
    max_epochs = args['n_train']
    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    save_path = args['save_path']
    n_nodes = args['n_nodes']
    dataGen = DataGenerator(args)
    data = dataGen.get_train_next()
    # print("train next: {}, size: {}".format(data[0], len(data)))
    # print("\n")

    data = dataGen.get_test_all()
    # print("test all: {}".format(data))
    # print("\n")

    # load GNN model
    gnn_model = get_model(args)

    env = Env(args, data)
    actor = Actor(hidden_size=args['hidden_dim'], num_layers=args['rnn_layers'],
                  dropout=args['dropout'], mask_logits=args['mask_logits'],
                  gnn_model=gnn_model)
    critic = Critic(hidden_size=args['hidden_dim'], num_layers=args['rnn_layers'])

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        path = save_path + 'n' + str(n_nodes) + '/best_model_actor_truck_params.pkl'
        if os.path.exists(path):
            actor.load_state_dict(torch.load(path, map_location='cpu'))
            path = save_path + 'n' + str(n_nodes) + '/best_model_critic_params.pkl'
            critic.load_state_dict(torch.load(path, map_location='cpu'))
            print("Successfully loaded keys")

    agent = A2CAgent(actor, critic, args, env, dataGen)
    if args['train']:
        agent.train()
    else:
        if args['sampling']:
            best_R = agent.sampling_batch(args['n_samples'])
        else:
            R = agent.test()

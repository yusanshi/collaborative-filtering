import argparse
from distutils.util import strtobool


def str2bool(x):
    return bool(strtobool(x))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--num_batches_show_loss', type=int, default=50)
    parser.add_argument('--num_epochs_validate', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--user_num', type=int, required=True)
    parser.add_argument('--item_num', type=int, required=True)
    parser.add_argument('--negative_sampling_ratio', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=3,
                        help='For GBPR')  # TODO: -1?
    parser.add_argument('--group_coefficient_rho',
                        type=float,
                        default=0.5,
                        help='For GBPR')
    parser.add_argument('--model_name',
                        type=str,
                        default='MF',
                        choices=['MF', 'MLP'])
    parser.add_argument('--loss_type',
                        type=str,
                        default='BCE',
                        choices=['BCE', 'CE', 'BPR', 'GBPR'])
    parser.add_argument(
        '--evaluation_metrics',
        type=str,
        nargs='+',
        default=['AUC', 'MRR', 'NDCG@10', 'NDCG@50', 'Recall@10', 'Recall@50'])
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--log_path', type=str, default='./log/')
    parser.add_argument('--tensorboard_runs_path', type=str, default='./runs/')
    args, unknown = parser.parse_known_args()
    if len(unknown) > 0:
        print('Warning: you have got some parameters wrong input')
    if args.loss_type in ['BPR', 'GBPR']:
        assert args.negative_sampling_ratio == 1, 'Negative sampling ratio must be 1 for BPR family loss'
    return args

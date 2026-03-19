import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='STRING')
    parser.add_argument('--cancerType', type=str, default='luad')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)

    parser.add_argument('--base_feature_dim', type=int, default=3)

    parser.add_argument('--feature_gate_hidden_dim', type=int, default=64)
    parser.add_argument('--topo_prompt_dim', type=int, default=16)
    parser.add_argument('--gate_hidden_dim', type=int, default=32)
    parser.add_argument('--num_pos_prototypes', type=int, default=3)
    parser.add_argument('--patience', type=int, default=35)

    parser.add_argument('--proto_alpha', type=float, default=0.25)
    parser.add_argument('--ema_momentum', type=float, default=0.95)
    parser.add_argument('--proto_shift_scale', type=float, default=0.15)
    parser.add_argument('--proto_logit_scale', type=float, default=0.20)
    parser.add_argument('--logit_temperature', type=float, default=2.5)

    parser.add_argument('--task_aug_ratio', type=float, default=0.20)
    parser.add_argument('--task_aug_start_epoch', type=int, default=50)

    parser.add_argument('--cls_pos_weight', type=float, default=8.0)
    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--lambda_proto', type=float, default=0.10)
    parser.add_argument('--lambda_shared', type=float, default=0.05)
    parser.add_argument('--lambda_gate_div', type=float, default=0.03)
    parser.add_argument('--lambda_task_aug', type=float, default=0.01)
    parser.add_argument('--lambda_gate_sparse', type=float, default=1e-3)
    parser.add_argument('--lambda_proto_balance', type=float, default=0.03)

    parser.add_argument('--episode_pos', type=int, default=12)
    parser.add_argument('--episode_neg', type=int, default=48)
    parser.add_argument('--ema_warmup_epochs', type=int, default=8)
    return parser.parse_args()

args = get_args()

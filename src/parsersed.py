#parser.py

import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='STRING')
    parser.add_argument('--cancerType', type=str, default='luad')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--base_feature_dim', type=int, default=3, help='number of base omics features')
    parser.add_argument('--feature_gate_hidden_dim', type=int, default=32, help='hidden dim of feature gate')

    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr',type=float, default=1e-4)
    parser.add_argument('--epochs' ,type=int, default=150)
    parser.add_argument('--n_splits', type=int, default=5)
    parser.add_argument('--patience', type=int, default=30)

    parser.add_argument('--use_diffusion_view', action='store_true')
    parser.add_argument('--diff_alpha', type=float, default=0.15)
    parser.add_argument('--diff_topk', type=int, default=50)

    parser.add_argument('--proto_alpha', type=float, default=0.25)
    parser.add_argument('--proto_topk_ratio', type=float, default=0.6)
    parser.add_argument('--min_proto_k', type=int, default=8)

    parser.add_argument('--cls_pos_weight', type=float, default=8.0)
    parser.add_argument('--lambda_cls', type=float, default=1.0)
    parser.add_argument('--lambda_proto', type=float, default=0.3)
    parser.add_argument('--lambda_hard_neg', type=float, default=0.4)
    parser.add_argument('--lambda_align', type=float, default=0.15)
    parser.add_argument('--lambda_mixup', type=float, default=0.15)

    parser.add_argument('--proto_margin', type=float, default=1.0)
    parser.add_argument('--hard_neg_margin', type=float, default=0.5)
    parser.add_argument('--hard_neg_topk', type=int, default=64)

    parser.add_argument('--mixup_num', type=int, default=32)
    parser.add_argument('--mixup_alpha', type=float, default=0.4)

    parser.add_argument('--hard_pos_threshold', type=float, default=0.7)
    parser.add_argument('--hard_pos_extra_weight', type=float, default=1.5)

    parser.add_argument('--align_max_nodes', type=int, default=256)
    args = parser.parse_args()
    return args


args = get_args()
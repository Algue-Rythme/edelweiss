import argparse
from utils import GridSearch

def get_grid_search_params(training_type, mode, yuqing_version, classifier, n_way, n_val, n_shot):
    grid = GridSearch()
    grid.add_range('training_type', [training_type])
    grid.add_range('mode', [mode])
    grid.add_range('n_way', [n_way])
    grid.add_range('n_val', [n_val])
    grid.add_range('n_shot', [n_shot])
    grid.add_range('yuqing_version', [yuqing_version])
    if mode not in ['yuqing_only', 'monitoring_volume']:
        # 640 - 300
        grid.add_range('n_feat', [320, 160, 80, 40, 20, 10, 5])
        grid.add_range('n_steps', [700, 1200, 1800, 2200, 2600, 3200, 3600])
        grid.add_group('n_feat', 'n_steps')
    if 'yuqing' in mode:
        grid.add_range('regular', [True])
        if mode != 'yuqing_only':
            grid.add_range('latent_diffused', ['x_latent', 'z_latent', 'z_latent'])
            grid.add_range('latent_graph_support', ['z_latent', 'x_latent', 'z_latent'])
            grid.add_group('latent_diffused', 'latent_graph_support')
        else:
            grid.add_range('frequency', [0.5, 1, 2, 4, 8])
            # grid.add_range('yuqing_version', ['pruned']) 
            # grid.add_range('n_fixpoint_graph', [5, 3, 1, 0])
        grid.add_range('num_neighbors', [3, 5, 10, 15, 20, 30, 40])
        grid.add_range('kappa', [1]) # [1, 2, 3]
        grid.add_range('alpha', [0.75])  # [1.25]
        grid.add_range('undirected', [True])
        grid.add_range('transposed_diffusion', [False])
        grid.add_range('svd_dim', [None])
        grid.add_range('inverse_svd', [False])
        grid.add_range('filter_name', ['heat'])
    if mode == 'monitoring_volume':
        grid.add_range('num_neighbors', [20])
        grid.add_range('regular', [False])
        grid.add_range('higher_order', [False])
        grid.add_range('kappa', [2]) # [1, 2, 3]
        grid.add_range('alpha', [0.75])  # [1.25]
        # grid.add_range('intersection_measure', ['minimum_cut'])
        # grid.add_range('intersection_measure', ['stoer_wagner'])
        # grid.add_range('intersection_measure', ['kernighan']) 
        # grid.add_range('intersection_measure', ['louvain_dendrogram'])
        # grid.add_range('intersection_measure', ['baseline'])
        grid.add_range('intersection_measure', ['arena'])
        grid.add_range('communities', ['entropy'])  # 'pure'
        grid.add_range('worse_only', [False])
        grid.add_range('crop', [False])  # 'pure'
        grid.add_range('ladder', [-1])  # 'pure'
        grid.add_range('dot_name', ['wideresnet/louvain_dendrogram_communities_1_20.dot'])
    grid.add_range('origin_normalization', ['mean-l2'])
    if 'orthonormal' in mode:
        grid.add_range('latent_normalization', ['l2'])
    elif 'JSD' in mode:
        grid.add_range('latent_normalization', ['l1'])
    elif 'yuqing' in mode:
        grid.add_range('latent_normalization', ['l2'])
    else:
        grid.add_range('latent_normalization', ['l2'])
    grid.add_range('classifier', [classifier])  # ['logistic_regression', 'ncm']
    grid.add_range('compute_corr', [True])
    grid.add_range('plot', [False])
    grid.add_range('progressive_plot', [False])
    return grid

def parse_args(modes):
    parser = argparse.ArgumentParser(description='Graph Smoothness.')
    parser.add_argument('--dataset', default='yuqing', help='Dataset key.')
    parser.add_argument('--mode', default='monitoring_volume', help='Mode of regularization. Can be %s'%modes)
    parser.add_argument('--yuqing_version', default='vanilla', help='Way to propagate features.')
    parser.add_argument('--classifier', default='logistic_regression', help='How to classify examples.')
    parser.add_argument('--n_way', default=2, type=int, help='number of classes.')
    parser.add_argument('--n_val', default=15, type=int, help='number of validation examples.')
    parser.add_argument('--n_shot', default=5, type=int, help='number of training examples.')
    sanity_check_parser = parser.add_mutually_exclusive_group(required=True)
    sanity_check_parser.add_argument('--sanity_check', dest='training_type', action='store_const', const='sanity_check')
    sanity_check_parser.add_argument('--brief_overview', dest='training_type', action='store_const', const='brief_overview')
    sanity_check_parser.add_argument('--full_training', dest='training_type', action='store_const', const='full_training')
    sanity_check_parser.add_argument('--test_baseline', dest='training_type', action='store_const', const='test_baseline')
    sanity_check_parser.add_argument('--single', dest='training_type', action='store_const', const='single')
    parser.set_defaults(training_type='full_training')
    return parser.parse_args()

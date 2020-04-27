import numpy as np
from tqdm import tqdm
import argparse
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
from classifiers import features_classification
from latent_graph import compute_smoothness, train_graph_embedder, embed_into_graph
from latent_graph import get_bad_edges, get_good_edges, get_inter_intra_edges 
from diffusion_graph import Combined_Yuqing_Myriam_Louis_Monster_Learnings, General_Yuqing
from loaders import get_train_test_datasets
from baseline_graph import BaselineModel, OrthoNormalModel, cosine_loss, inter_intra_loss, cov_loss
from jsd_graph import SomewhatStochasticModel, jsd_loss
from garbage import OldInfoRecorder
from monitoring import monitore_volume, monitore_communities, monitore_regression, monitore_arena
from parse_grid import get_grid_search_params, parse_args


def get_model(num_features, params):
    if 'JSD' in params.mode:
        return SomewhatStochasticModel(num_features, params.n_feat)
    if 'baseline' in params.mode:
        return BaselineModel(num_features, params.n_feat)
    if 'orthonormal' in params.mode:
        return OrthoNormalModel(num_features, params.n_feat)
    assert False

def get_loss_fn(params):
    if 'JSD' in params.mode:
        return jsd_loss
    if 'inter-intra' in params.mode:
        return inter_intra_loss
    if params.mode == 'yuqing_only' and params.transposed_diffusion:
        return cov_loss
    return cosine_loss  # default similarity measure

def get_edge_selector(mode):
    if 'inter-intra' in mode:
        return get_inter_intra_edges
    return get_bad_edges

def compute_smoothness_accuraccy(recorder, train_set, test_set, train_labels, test_labels, params):
    if params.training_type == 'test_baseline':
        infos_latent = (-1., -1.)
    else:
        infos_latent = features_classification(train_set, train_labels, test_set, test_labels, n_way,
                                               params.classifier, params.latent_normalization, params)
    if params.compute_smoothness:
        loss_fn = get_loss_fn(params)
        edge_selector = get_edge_selector(params.mode)
        train_smooth = compute_smoothness(train_set, train_labels, loss_fn, edge_selector)
        test_smooth = compute_smoothness(test_set, test_labels, loss_fn, edge_selector)
        recorder.record_infos_smooth(train_smooth, test_smooth)
    recorder.record_info_results(infos_latent)

def reduce_smoothness(x_train, x_test, train_labels, test_labels, loss_fn, params):
    num_features = int(x_train.shape[1])
    model = get_model(num_features, params)
    edge_selector = get_edge_selector(params.mode)
    model = train_graph_embedder(x_train, train_labels, model, loss_fn, edge_selector, params.n_steps)
    if 'yuqing' in params.mode:
        return Combined_Yuqing_Myriam_Louis_Monster_Learnings(x_train, x_test,
                                                              model, loss_fn,
                                                              params)
    z_train = embed_into_graph(model, x_train)
    z_test = embed_into_graph(model, x_test)
    return z_train, z_test

def launch_experiment(recorder, data_path, params):
    train_test = get_train_test_datasets(data_path, params.n_way, params.n_shot, params.n_val)
    train_set, train_labels, test_set, test_labels = train_test
    infos_origin = features_classification(train_set, train_labels,
                                           test_set, test_labels,
                                           params.n_way, params.classifier, params.origin_normalization, params)
    recorder.record_origin(infos_origin)
    if params.training_type == 'test_baseline':
        return
    loss_fn = get_loss_fn(params)
    if params.mode == 'monitoring_volume':
        with torch.no_grad():
            monitore_volume(recorder, train_set, test_set, train_labels, test_labels, params)
            return
    elif params.mode == 'yuqing_only':
        train_set, test_set = General_Yuqing(recorder, train_set, test_set, train_labels, loss_fn, params)
    else:
        train_set, test_set = reduce_smoothness(train_set, test_set, train_labels, test_labels, loss_fn, params)
    compute_smoothness_accuraccy(recorder, train_set, test_set,
                                 train_labels, test_labels,
                                 params)

def add_point(fig, sc, recorder, params, vars_corr):
    recorder_dict = recorder.get_dict()
    test_orig, test_acc = recorder_dict[vars_corr[0][0]], recorder_dict[vars_corr[0][1]]
    sc.set_offsets(np.c_[test_orig, test_acc])
    if params.plot:
        fig.canvas.draw_idle()
        plt.pause(0.1)

def init_plot(params):
    if params.plot:
        plt.ion()
    fig, ax = plt.subplots()
    plt.xlim(60, 95)
    plt.ylim(60, 95)
    plt.plot([60, 95], [60, 95], c='orange', marker=',', linestyle='-')
    sc = ax.scatter([], [], marker='.')
    if params.plot:
        fig.canvas.draw_idle()
        plt.pause(0.1)
    return fig, sc

def misc_printing(recorder, params):
    if params.mode == 'monitoring_volume' and params.intersection_measure == 'louvain_dendrogram':
        return ''
    train_origin, test_origin = recorder.get_train_test_origin()
    if params.mode == 'monitoring_volume':
        volume_error = recorder.get_avg_volume_errors()
        desc = 'avg_test_origin=%.2f%% avg_volume_error=%.2f%%'%(test_origin, volume_error)
        # balance = ' last_balance=%.2f%%'%recorder.get_last_balance()
        last_volume = ' last_volume_error=%.2f%%'%(recorder.get_last_volume_error())
        last_acc = ' last_acc=%.2f%%'%(recorder.get_last_train_test_origin()[1])
        desc += (last_volume + last_acc)
        return desc
    desc = 'train_origin=%.2f%% test_origin=%.2f%%'%(train_origin, test_origin)
    if params.training_type != 'test_baseline':
        train_avg, test_avg = recorder.get_avg_train_test_acc()
        desc += ' train_avg=%.2f%% test_avg=%.2f%% delta=%.2f%%'%(train_avg, test_avg, test_avg - test_origin)
    if params.yuqing_version == 'pruned':
        desc += ' cut=%.2f%%'%recorder.get_cut()
    if params.yuqing_version == 'fourrier':
        desc += ' error=%.2f'%recorder.get_error()
    return desc

def compute_correlation(n_dataset, data_path, params, vars_corr, param_num):
    recorder = OldInfoRecorder()
    progress = tqdm(total=n_dataset, leave=True, desc='number of datasets')
    if params.progressive_plot:
        fig, sc = init_plot(params)
    for _ in range(n_dataset):
        launch_experiment(recorder, data_path, params)
        desc = misc_printing(recorder, params)
        if params.progressive_plot:
            add_point(fig, sc, recorder, params, vars_corr)
        progress.set_description(desc=desc)
        progress.update()
    if params.progressive_plot:
        fig.canvas.draw_idle()
        fig.savefig(os.path.join('graphs', str(param_num)+'.png'))
        plt.close(fig)
    progress.close()
    if params.compute_corr:
        results = recorder.get_dict()
        for var_corr in vars_corr:
            var_a = results[var_corr[0]]
            var_b = results[var_corr[1]]
            if len(var_a) == 1 or len(var_b) == 1:
                break
            corrcoefmatrix = np.corrcoef(var_a, var_b)
            print(var_corr, ' mean_corr=%.2f%%'%corrcoefmatrix[0,1], flush=True)
    print('')

def correlation_grid_search(n_dataset, data_path, grid_search_params):
    vars_corr = [('test_acc_orig', 'volume_error')]
                # ('test_acc_orig', 'balance'),
                # ('balance', 'volume_error')]
    print(grid_search_params.get_constant_keys())
    print('')
    for param_num, params in enumerate(grid_search_params.get_params()):
        if params.training_type != 'test_baseline':
            print(grid_search_params.get_variable_keys(params))
        measures = ['louvain_dendrogram', 'baseline', 'arena']
        if params.mode == 'monitoring_volume' and params.intersection_measure in measures:
            if params.intersection_measure == 'louvain_dendrogram':
                monitore_communities(data_path, params)
            elif params.intersection_measure == 'baseline':
                monitore_regression(data_path, params)
            elif params.intersection_measure == 'arena':
                with torch.no_grad():
                    monitore_arena(data_path, params)
        else:
            compute_correlation(n_dataset, data_path, params, vars_corr, param_num)
        if params.training_type == 'test_baseline':
            break

def get_num_tests(training_type):
    num_tests = dict()
    num_tests['full_training'] = 1000
    num_tests['brief_overview'] = 100
    num_tests['sanity_check'] = 2
    num_tests['test_baseline'] = 10 * 1000
    num_tests['single'] = 1
    return num_tests[training_type]

if __name__ == '__main__':
    modes = ['JSD', 'baseline', 'orthonormal']
    modes += [mode+'_inter-intra' for mode in modes]
    modes += [mode+'_yuqing' for mode in modes] + ['yuqing_only'] + ['monitoring_volume']
    args = parse_args(modes)
    n_dataset = get_num_tests(args.training_type)
    data_paths = {'vanilla':'images/latent/miniImagenet/ResNet/layer5/features.pt',
                  'yuqing':'images/latent/miniImagenet/WideResNet28_10_S2M2_R/last/novel.plk',
                  'cross':'images/latent/cross/WideResNet28_10_S2M2_R/last/output.plk'}
    data_path = data_paths['yuqing']
    n_way = args.n_way
    n_val = args.n_val
    n_shot = args.n_shot
    to_print = args.mode, args.yuqing_version, args.classifier, n_way, n_val, n_shot, data_path
    print('mode=%s yuqing_version=%s classifier=%s n_way=%d n_val=%d n_shot=%d datapath=%s'%to_print)
    if args.mode in modes:
        grid_search_params = get_grid_search_params(args.training_type, args.mode, args.yuqing_version,
                                                    args.classifier, n_way, n_val, n_shot)
        correlation_grid_search(n_dataset, data_path, grid_search_params)
    else:
        print('Bad experience name')

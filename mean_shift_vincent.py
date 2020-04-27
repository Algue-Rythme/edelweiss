import pickle
import numpy as np
import torch
import math

# Parsing arguments, should be quite self-explanatory
import argparse
parser = argparse.ArgumentParser(description='Few-Shot Transfer')
parser.add_argument('--ways', type=int, default="5")
parser.add_argument('--shots', type=int, default="1")
parser.add_argument('--q', type=int, default="15")
parser.add_argument('--runs', type=int, default="10000")
parser.add_argument('--dataset', type=str, default="miniimagenet")
parser.add_argument('--noncautious', action="store_true", help="use standard mean-shift")
parser.add_argument('--m1', type=float, default="0.8")
parser.add_argument('--m2', type=float, default="0")
parser.add_argument('--spring', type=float, default="0")

# Arguments with respect to the graph creation
parser.add_argument('--beta', type=float, default="1")
parser.add_argument('--alpha', type=float, default="3")
parser.add_argument('--p', type=int, default="3")
parser.add_argument('--k', type=int, default="5")
parser.add_argument('--normalized', action="store_true")

args = parser.parse_args()

# Function to display results not too frequently
import sys
import time
last_tick = 0
def display(s, force = False):
    global last_tick
    if time.time() - last_tick > 0.5 or force:
        sys.stderr.write(s)
        last_tick = time.time()

# Loading data from files on computer
from os.path import expanduser
home = expanduser("~")
def load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        labels = [np.full(shape=len(data[key]), fill_value=key) for key in data]
        data = [features for key in data for features in data[key]]
        dataset = dict()
        dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
        dataset['labels'] = torch.LongTensor(np.concatenate(labels))
        return dataset

if args.dataset == "CUB":
    dataset = load_pickle(home + "/datasets/WideResNet28_10/CUB-cross.pkl")
elif args.dataset == "miniimagenet":
    dataset = load_pickle(home + "/datasets/WideResNet28_10/output.plk")

# Computing the number of items per class in the dataset
min_examples = dataset["labels"].shape[0]
for i in range(dataset["labels"].shape[0]):
    if torch.where(dataset["labels"] == dataset["labels"][i])[0].shape[0] > 0:
        min_examples = min(min_examples, torch.where(dataset["labels"] == dataset["labels"][i])[0].shape[0])
#display("Guaranteed number of items per class: {:d}\n".format(min_examples), True)

# Generating data tensors
data = torch.zeros((0,min_examples,dataset["data"].shape[1]))
labels = dataset["labels"].clone()
while labels.shape[0] > 0:
    indices = torch.where(dataset["labels"] == labels[0])[0]
    data = torch.cat([data, dataset["data"][indices,:][:min_examples].view(1,min_examples, -1)], dim = 0)
    indices = torch.where(labels != labels[0])[0]
    labels = labels[indices]
#display("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(data.shape[0], data.shape[1], data.shape[2]), True)

# Function to generate random tasks
shuffle_indices = np.arange(min_examples)
def shuffle():
    global data, shuffle_indices    
    for i in range(data.shape[0]):
        shuffle_indices = np.random.permutation(shuffle_indices)
        data[i,:,:] = data[i,shuffle_indices,:]

# First magic trick
data = torch.pow(data,0.5)
# Sphere projection
data = data / torch.norm(data, dim=2, keepdim = True)

# simple ncm classifier for baseline
# apparently, Euclidean and cosine are about as accurate
def ncm(dataset, cosine = False):
    means = dataset[:,:args.shots,:].mean(dim=1)
    if cosine: #project to unit hypershere
        means = means / torch.norm(means, dim = 1, keepdim=True)
    if not cosine:
        dist = torch.norm(dataset[:,args.shots:,:].view(args.ways, args.q, 1, -1) - means.view(1,1,means.shape[0], means.shape[1]), dim = 3, p = 2) #dist is ways x queries x centroids
        _, decisions = torch.min(dist, dim=2)
    else:
        dist = torch.matmul(dataset[:,args.shots:,:],means.transpose(0,1))
        _, decisions = torch.max(dist, dim=2)
    return (decisions == torch.LongTensor((np.arange(args.ways))).unsqueeze(1)).float().mean().item()

### mean-shift classifier
# aux function to reset probas of labeled elements
def oracle(probas):
    for i in range(args.ways):
        probas[np.arange(args.shots) + (args.q + args.shots) * i] = 0
        probas[np.arange(args.shots) + (args.q + args.shots) * i,i] = 1
        
# main mean-shift function
def mean_shift(dataset, cautious = not args.noncautious):
    flatten_dataset = dataset.reshape(-1, dataset.shape[2])
    # vectors of probas for each element to belong to each class
    probas = torch.zeros(((args.q + args.shots) * args.ways, args.ways))
    oracle(probas)
    old_probas = torch.zeros_like(probas)
    # iterations begin
    for step in range(100):
        # first we estimate centroids
        if step == 0:
            centroids = ((probas / torch.sum(probas, dim = 0, keepdim=True)).view(-1, 1, args.ways) * flatten_dataset.view(-1, flatten_dataset.shape[1], 1)).sum(dim = 0).transpose(0,1)
        elif step < 50:
            centroids = args.m1 * centroids + (1-args.m1) * ((probas / torch.sum(probas, dim = 0, keepdim=True)).view(-1, 1, args.ways) * flatten_dataset.view(-1, flatten_dataset.shape[1], 1)).sum(dim = 0).transpose(0,1)
        else:
            centroids = args.m2 * centroids + (1-args.m2) * ((probas / torch.sum(probas, dim = 0, keepdim=True)).view(-1, 1, args.ways) * flatten_dataset.view(-1, flatten_dataset.shape[1], 1)).sum(dim = 0).transpose(0,1)
        for i in range(args.ways):
            for j in range(args.ways):
                centroids[i]-=args.spring*(centroids[j]-centroids[i])
        # then we compute distances
        dist = torch.norm(flatten_dataset.view(-1,1,flatten_dataset.shape[1]) - centroids.view(1,args.ways,flatten_dataset.shape[1]), dim = 2, p = 2)
        # convert to similarities
        sim = torch.exp(-10 * torch.pow(dist, 2))
        # normalize to sum to 1 for each class
        n_sim = sim / torch.norm(sim, dim = 1, p = 1, keepdim=True)
        # we get predictions
        _, decisions = torch.max(n_sim, dim = 1)
        if cautious:
            # find smallest cluster and be sure we do not go too fast
            minimum = min(np.min([(decisions == i).int().sum().item() for i in np.arange(args.ways)]), args.shots + step + 1)
        else:
            # or just find the smallest cluster
            minimum = np.min([(decisions == i).int().sum().item() for i in np.arange(args.ways)])
        # find probas
        probas[:] = 0
        if step < 50:
            for i in range(args.ways):
                indices_i = torch.where(decisions == i)[0]
                scores = n_sim[indices_i]
                values, indices = torch.sort(scores[:,i], descending = True)
                probas[indices_i[indices[:minimum]],:] = 0
                probas[indices_i[indices[:minimum]],i] = 1#values[:minimum]
        else:
            for i in range(args.ways):
                values, indices = torch.sort(n_sim[:,i], descending = True)
                probas[indices[:args.q+args.shots],:] = 0
                probas[indices[:args.q+args.shots],i] = 1#values[:minimum]
                
        oracle(probas)
        # if (probas == old_probas).all():
        #     break
        old_probas = probas.clone()
    # Finally compute score
    targets = torch.LongTensor(np.asarray([[i] * (args.q + args.shots) for i in range(args.ways)]).flatten())
    for i in range(args.ways):
        decisions[np.arange(args.shots) + (args.q + args.shots) * i] = i
    return ((targets == decisions).float().sum().item() - (args.ways * args.shots)) / (args.ways * args.q)

# Diffuse data using graphs
# def diffusion(dataset):
#     res = torch.norm(dataset.view(-1,1,dataset.shape[1]) - dataset.view(1,-1,dataset.shape[1]), dim = 2, p = 2)
#     res = torch.exp(-args.beta * torch.pow(res, 2))
#     for i in range(res.shape[0]):
#         res[i,i] = 0

#     if args.k > 0:
#         y, ind = torch.sort(res, 1)
#         k_adj_matrix = torch.zeros(*y.size())
#         k_biggest = ind[:, -args.k:].data
#         for index1, value in enumerate(k_biggest):
#             adj_line = k_adj_matrix[index1]
#             adj_line[value] = 1
#         k_adj_symmetric = torch.min(
#            torch.ones(*y.size()), k_adj_matrix+torch.t(k_adj_matrix))
#         res = res*k_adj_matrix

#     if args.normalized:
#         degree_vector = torch.sum(res, 1)
#         degree_matrix = torch.diag(degree_vector)
#         laplacian = (degree_matrix - res)
#         degreehalf = torch.pow(degree_vector, -0.5)
#         degreehalf = torch.diag(degreehalf)
#         res = torch.mm(degreehalf, torch.mm(laplacian, degreehalf))

#     for i in range(res.shape[0]):
#         res[i,i] = args.alpha

#     new_res = res.clone()
#     for i in range(args.p - 1):
#         new_res = torch.matmul(new_res, res)
        
#     dataset = torch.matmul(new_res, dataset)
    
#     return dataset

def confidence(values):
    return np.std(values) * 1.96 / math.sqrt(len(values))


# main routine
baseline_score = []
for t in range(args.runs):
    # create task
    shuffle()
    classes = np.random.permutation(np.arange(data.shape[0]))[:args.ways]
    dataset = data[classes,:args.q + args.shots,:]

    # magic trick 1
    for i in range(3):
        dataset[:,:args.shots,:] = dataset[:,:args.shots,:] - torch.mean(torch.mean(dataset[:,:args.shots,:], dim=1, keepdim=True), dim=0, keepdim=True)
        dataset[:,:args.shots,:] = dataset[:,:args.shots,:] / torch.norm(dataset[:,:args.shots,:], dim=2, p=2, keepdim=True)
        dataset[:,args.shots:,:] = dataset[:,args.shots:,:] - torch.mean(torch.mean(dataset[:,args.shots:,:], dim=1, keepdim=True), dim=0, keepdim=True)
        dataset[:,args.shots:,:] = dataset[:,args.shots:,:] / torch.norm(dataset[:,args.shots:,:], dim=2, p=2, keepdim=True)
    
    # reshape for normalization
    base = dataset.reshape(-1, dataset.shape[2])
    
    # magic trick 2 (exclusive)
    for i in range(0):
        base = base - base.mean(dim=0)
        base = base / torch.norm(base, dim=1, p=2, keepdim=True)

    # Gram-Schidt reduction
    base = base.transpose(0,1)
    q_m,r = base.qr()
    dataset = torch.matmul(base.transpose(0,1), q_m)

    # dataset = diffusion(dataset)

    dataset = dataset.reshape(args.ways,-1 ,r.shape[0])
    
    baseline_score.append(mean_shift(dataset))
    # baseline_score.append(ncm(dataset))
    
    display("\rrun {:5d}/{:5d}: {:5f} {:5f}".format(t+1, args.runs, np.mean(baseline_score), confidence(baseline_score)), t + 1 == args.runs)

#print("\rdataset: {:s}, alpha: {:f}, beta: {:f}, p: {:d}, k: {:d}, normalized: {:b}, ways: {:d}, shots: {:d}, q: {:d}, runs: {:d}, mean: {:f}, std: {:f}".format(args.dataset, args.alpha, args.beta, args.p, args.k, args.normalized, args.ways, args.shots, args.q, args.runs, np.mean(baseline_score), confidence(baseline_score)))

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from latent_graph import PiModel

class BaselineModel(PiModel):

    def __init__(self, num_features, latent_dim):
        super(BaselineModel, self).__init__()
        self.kernel = nn.Linear(num_features, latent_dim)

    def forward(self, x):
        return self.kernel(x)


class OrthoNormalModel(PiModel):

    def __init__(self, num_features, latent_dim):
        super(OrthoNormalModel, self).__init__()
        self.kernel = nn.Parameter(data=torch.zeros(size=(latent_dim, num_features), 
                                                    dtype=torch.float32),
                                   requires_grad=True)
        nn.init.xavier_normal_(self.kernel)
        self.lbda = 1.
        self.epsilon = 1e-2

    def get_opt(self):
        return Adam(self.parameters(), lr=1e-2)

    def P(self):
        return nn.functional.relu(self.kernel)

    def regularization_loss(self):
        M = torch.mm(self.P(), torch.t(self.P()))
        Id = torch.eye(M.shape[0])
        delta = torch.norm(M - Id)
        loss = self.lbda * delta
        return loss

    def forward(self, x):
        x = x / (self.epsilon + torch.norm(x, dim=-1, keepdim=True)) # unit-sphere
        z = torch.einsum('ij,kj->ki', self.P(), x)
        return z

def cosine_loss(z_latent, bad_edges, reduce=True, memory_light=False):
    epsilon = 1e-2
    scalars = torch.einsum('ik,jk->ij', z_latent, z_latent)
    norms = epsilon + torch.norm(z_latent, dim=-1, keepdim=False)
    norms = torch.einsum('i,j->ij', norms, norms)
    cosines = scalars / norms
    losses = []
    for (a, b) in bad_edges:
        delta = cosines[a,b]
        if memory_light:
            losses.append(float(delta.item()))
        else:
            losses.append(delta)
    if reduce:
        return torch.mean(torch.stack(losses))
    return losses

def inter_intra_loss(z_latent, inter_intra_edges):
    bad_edges, good_edges = inter_intra_edges
    bad_edges_loss = cosine_loss(z_latent, bad_edges, reduce=False)
    good_edges_loss = cosine_loss(z_latent, good_edges, reduce=False)
    bad_edges_loss = torch.sum(torch.stack(bad_edges_loss))
    good_edges_loss = torch.sum(torch.stack(good_edges_loss))
    ratio = bad_edges_loss / good_edges_loss
    return ratio

def cov_loss(z_latent, bad_edges, reduce=True):
    z_latent = z_latent.numpy() # N x F
    corrmatrix = np.corrcoef(z_latent, rowvar=True)  # N x N output
    corrmatrix = np.abs(corrmatrix)  # abs(correlation) VERSUS independance (=0)
    corrmatrix = torch.FloatTensor(corrmatrix)
    losses = []
    for (a, b) in bad_edges:
        losses.append(corrmatrix[a,b])
    if reduce:
        return torch.mean(torch.stack(losses))
    return losses


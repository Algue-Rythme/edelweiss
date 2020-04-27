import torch
import torch.nn as nn
from torch.optim import Adam
from utils import get_device
from latent_graph import PiModel

class SomewhatStochasticModel(PiModel):

    def __init__(self, num_features, latent_dim):
        super(SomewhatStochasticModel, self).__init__()
        self.kernel = nn.Parameter(data=torch.zeros(size=(latent_dim, num_features), 
                                                    dtype=torch.float32),
                                   requires_grad=True)
        nn.init.xavier_normal_(self.kernel)
        self.softmax = nn.Softmax(dim=0)
        self.init_temperature = 3e-2
        self.exponent = 1.1
        self.max_exp = 50.
        self.temperature = self.init_temperature

    def get_opt(self):
        return Adam(self.parameters(), lr=1e-1)

    def init_epoch(self, n_steps):
        self.n_steps = n_steps

    def update_step(self, step):
        ratio = step / self.n_steps
        exp_part = self.exponent ** (ratio * self.max_exp)
        self.temperature = 1. #self.init_temperature * exp_part

    def init_inference(self):
        self.update_step(self.n_steps)

    def forward(self, x):
        """Forward on batch of shape (n_samples, input_dim).

        Returns:
            Batch of examples in latent space Z of shape (n_samples, latent_dim)
        """
        somewhat = self.softmax(self.kernel * self.temperature)
        x = x / torch.sum(x, dim=1, keepdim=True)
        z = torch.einsum('ij,kj->ki', somewhat, x)
        return z


def kl_div(p, support):
    return (p * (p / support).log()).sum(dim=-1)

def JSD(p, q):
    mixed = 0.5*p + 0.5*q
    jsd = kl_div(p, mixed) + kl_div(q, mixed)
    return jsd

def jsd_loss(z_latent, bad_edges, reduce=True):
    losses = []
    for (a, b) in bad_edges:
        p = z_latent[a,:]
        q = z_latent[b,:]
        loss = JSD(p, q)
        losses.append(-loss) # behold the minus for maximization of distance
    if reduce:
        return torch.mean(torch.stack(losses))
    return losses 
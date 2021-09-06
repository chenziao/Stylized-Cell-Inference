import torch
import pyro
from torch.distributions import Normal, Uniform
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape


class StackedDistribution(TorchDistribution):
    def __init__(
        self,
        base_dists,
        validate_args=False
    ):
        assert len(set(base_dist.event_shape for base_dist in base_dists)) == 1, \
            "All base distributions should have the same event shape"
        batch_shape = broadcast_shape([base_dist.batch_shape for base_dist in base_dists])
        bs = [1 for i in batch_shape] #done because each distribution is 1D and torch.size flattens this
        event_shape = base_dists[0].event_shape + (len(base_dists),)
        self.base_dists = tuple(base_dist.expand(bs) for base_dist in base_dists)
        super().__init__(batch_shape, event_shape, validate_args)
    
    def log_prob(self, x):
        return sum(base_dist.log_prob(x[..., i]) for i, base_dist in enumerate(self.base_dists))

    def rsample(self, sample_shape=torch.Size()):
        return torch.stack([base_dist.rsample(sample_shape) for base_dist in self.base_dists], dim=-1)

    def sample(self, sample_shape=torch.Size()):
        return torch.stack([base_dist.rsample(sample_shape) for base_dist in self.base_dists], dim=-1)

    @property
    def has_rsample(self):
        return all(base_dist.has_rsample for base_dist in self.base_dists)


def build_priors(bounds):
    priors = []
    for k, v in bounds:
        if k == 'U':
            priors.append(Uniform(v[0],v[1]))
        elif k == 'N':
            priors.append(Normal(v[0],v[1]))
    box_dist = StackedDistribution(priors)
    return box_dist

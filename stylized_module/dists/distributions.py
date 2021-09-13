import torch
import numpy as np
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape


class StackedDistribution(TorchDistribution):
    def __init__(
        self,
        base_dist1,
        base_dist2,
        validate_args=False
    ):
        # print(base_dists)
        assert base_dist1.event_shape == base_dist2.event_shape, \
            "Both base distributions should have the same event shape"
        batch_shape = torch.from_numpy(np.array(broadcast_shape(base_dist1, base_dist2)))
        event_shape = base_dist1[0].event_shape
        self.base_dists = [base_dist1.expand(batch_shape), base_dist2.expand(batch_shape)]
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

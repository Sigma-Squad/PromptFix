import torch
import numpy as np


class DiagonalGaussianDistribution(object):

    def __init__(self, parameter: torch.Tensor, deterministic: bool = False) -> None:
        self.parameter = parameter
        self.mean, self.logvar = torch.chunk(parameter, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5*self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean).to(self.parameter.device)

    def sample(self) -> torch.Tensor:
        if self.deterministic:
            return self.mean
        else:
            return self.mean + self.std * torch.randn_like(self.std)

    def kl(self, other: torch.Tensor = None) -> torch.Tensor:
        if self.deterministic:
            return torch.tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - self.logvar - 1, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - self.logvar + other.logvar - 1, dim=[1, 2, 3])

    def nll(self, sample: torch.Tensor, dims: list = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.tensor([0.0])
        return 0.5*torch.sum(torch.pow(sample-self.mean, 2)/self.var + self.logvar + np.log(2.0*np.pi), dim=dims)

    def mode(self) -> torch.Tensor:
        return self.mean

import torch
import torch.nn as nn
from .util import init
import numpy as np
"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions) #.sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy() #.sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, action_space, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(DiagGaussian, self).__init__()

        self.hidden_size = 64
        # init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        # def init_(m): 
        #     return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
            
        self.fc_mean = nn.Sequential(
            # nn.Linear(num_inputs, num_outputs)
            nn.Linear(num_inputs, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_outputs),
            nn.Tanh(),
        )    
        self.fc_std = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_outputs),
        )
        action_range = (action_space.high-action_space.low) / 2
        self.action_range = torch.ones(num_outputs) * action_range
        action_mid = (action_space.high+action_space.low) / 2
        self.action_mid = torch.ones(num_outputs) * action_mid
        self.logstd = AddBias(torch.log(self.action_range))

    def forward(self, x):
        self.action_range = self.action_range.to(x.device)
        self.action_mid = self.action_mid.to(x.device)
        action_mean = self.fc_mean(x) * self.action_range + self.action_mid
        action_logstd = self.fc_std(x) 
        action_std = torch.clamp(action_logstd, -10, 2).exp()
        # zeros = torch.zeros(action_mean.size())
        # if x.is_cuda:
        #     zeros = zeros.cuda()
        # action_logstd = self.logstd(zeros)
        # action_logstd_bias = self.fc_std(x) # * self.action_range
        # action_stdbias = torch.clip(action_logstd_bias.exp(),0.5,1.5)
        # action_std = action_logstd.exp() * action_stdbias
        return FixedNormal(action_mean, action_std)


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

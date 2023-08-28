import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, pixel=False):
        super(Policy, self).__init__()
        self.pixel = pixel
        if self.pixel:
            self.resnet = resnet18(num_classes=100)
        self.affine1 = nn.Linear(num_inputs, 16)
        self.affine2 = nn.Linear(16, 16)

        self.action_mean = nn.Linear(16, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        if self.pixel:
            x = x.reshape(-1, 3, 32, 32)
            x = self.resnet(x)
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class QValue(nn.Module):
    def __init__(self, state_size, action_size):
        super(QValue, self).__init__()
        self.action_size = action_size
        num_inputs = state_size + action_size
        hidden_dim = 64
        self.affine1 = nn.Linear(num_inputs, hidden_dim)
        self.affine2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values

    def get_value(self, state, policy):
        batch_size = len(state)
        with torch.no_grad():
            action_mean, _, action_std = policy(state)
            # eps = torch.randn(batch_size * 10 ,self.action_size)
            # print(action_mean.shape)
            # print(action_std.shape)
            # print(eps.shape)
        #     actions = action_mean.repeat(10,1) + action_std.repeat(10,1) * eps
        #     states = state.repeat(10,1)

        # return self.forward(torch.cat([states,actions],dim=-1)).reshape(10,batch_size,-1).mean(0)

            value = torch.zeros(batch_size, 1).to(state.device)
            for _ in range(10):
                action = torch.normal(action_mean, action_std)
                value = value + \
                    self.forward(torch.cat([state, action], dim=-1))

        return value / 10

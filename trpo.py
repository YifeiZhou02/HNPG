import numpy as np

import torch
from torch.autograd import Variable
from utils import *
from models import *
# FINITE flag is not supported after I split the value network
# only need to change this flag to switch to splitted policy
FINITE = True
# change this flag to switch from vanilla trpo to hybrid trpo
HYBRID = True


def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size()).to(b.device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(model,
               f,
               x,
               fullstep,
               expected_improve_rate,
               max_backtracks=10,
               accept_ratio=.1):
    fval = f(True).data
    # print("fval before", fval.item())
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        # print("a/e/r", actual_improve.item(),
        #       expected_improve.item(), ratio.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            # print("fval after", newfval.item())
            return True, xnew, fval.item(), newfval.item()
    return False, x, fval.item(), newfval.item()


def trpo_step(model, get_loss, get_kl, max_kl, damping, device):
    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data.to(device)

    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat(
            [grad.contiguous().view(-1) for grad in grads]).data.to(device)

        return flat_grad_grad_kl + v * damping

    stepdir = conjugate_gradients(Fvp, -loss_grad, 10)

    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)

    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdotstepdir = (-loss_grad * stepdir).sum(0, keepdim=True)
    # print(("lagrange multiplier:", lm[0], "grad_norm:", loss_grad.norm()))

    prev_params = get_flat_params_from(model)
    success, new_params, loss_before, loss_after = linesearch(model, get_loss, prev_params, fullstep,
                                                              neggdotstepdir / lm[0])
    set_flat_params_to(model, new_params)

    return loss_before, loss_after


def horizon_get_actions(states, horizons, policy_nets, device):
    action_means = []
    action_log_stds = []
    action_stds = []
    for i, h in enumerate(horizons):
        acm, acl, acs = policy_nets[int(h)](
            Variable(states[i].unsqueeze(0)).to(device))
        action_means.append(acm)
        action_log_stds.append(acl)
        action_stds.append(acs)
    action_means = torch.cat(action_means, dim=0)
    action_log_stds = torch.cat(action_log_stds, dim=0)
    action_stds = torch.cat(action_stds, dim=0)
    return action_means, action_log_stds, action_stds


def horizon_get_values(states, horizons, policy_nets, q_net, device):
    values = []
    for i, h in enumerate(horizons):
        values.append(q_net[h].get_value(
            states[i].unsqueeze(0).to(device), policy_nets[int(h)]))
    return torch.cat(values, dim=0)


def update_policy(policy_net, states, actions, advantages, max_kl, damping, device):
    # off_advantages = (off_advantages - off_advantages.mean()
    #   ) / off_advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(
        Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(
                    Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(
                Variable(states))

        log_prob = normal_log_density(
            Variable(actions), action_means, action_log_stds, action_stds)

        # add offline states

        action_loss = (-Variable(advantages) *
                       torch.exp(log_prob - Variable(fixed_log_prob))).mean()

        return action_loss.mean()

    def get_kl():
        mean1, log_std1, std1 = policy_net(
            Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + \
            (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    loss_before, loss_after = trpo_step(
        policy_net, get_loss, get_kl, max_kl, damping, device)
    return loss_before, loss_after


class TRPO(object):
    def __init__(self, num_states, num_actions, horizon, l2_reg, gamma, tau, max_kl, damping,
                 critic_ratio, device=torch.device('cpu')):
        if FINITE:
            self.policy_net = [Policy(num_states, num_actions).to(device)
                               for _ in range(horizon)]
        else:
            self.policy_net = Policy(num_states, num_actions)
        self.q_net = [QValue(num_states, num_actions).to(device)
                      for _ in range(horizon)]

        self.l2_reg = l2_reg
        self.gamma = 1
        self.tau = tau
        self.damping = damping
        self.max_kl = max_kl

        self.critic_ratio = critic_ratio
        self.device = device

    def select_action(self, state, h):
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        if FINITE:
            action_mean, _, action_std = self.policy_net[h](Variable(state))
        else:
            action_mean, _, action_std = self.policy_net(Variable(state))
        action = torch.normal(action_mean, action_std)
        return action.cpu()

    def update_q(self, states, actions, targets, off_states, off_actions, off_targets, q_net):
        # set_flat_params_to(q_net, torch.Tensor(flat_params))
        optimizer = torch.optim.Adam(q_net.parameters(), lr=3e-4)

        # print(len(states))
        # print(len(off_states))
        online_stateactions = torch.cat(
            [states, actions], dim=1).to(self.device)
        targets = targets.to(self.device)
        off_stateactions = torch.cat(
            [off_states, off_actions], dim=1).to(self.device)
        off_targets = off_targets.to(self.device)
        for _ in range(30):
            optimizer.zero_grad()
            qs_ = q_net(online_stateactions)
            on_q_loss = (qs_ - targets).pow(2).mean()

            for param in q_net.parameters():
                q_loss = on_q_loss + param.pow(2).sum() * self.l2_reg

            off_qs_ = q_net(off_stateactions)
            off_b_loss = (off_qs_-off_targets).pow(2).mean()
            if HYBRID:
                q_loss += off_b_loss
            q_loss.backward()

            optimizer.step()
        # for param in self.q_net.parameters():
        #     if param.grad is not None:
        #         param.grad.data.fill_(0)
        # print(Variable(torch.cat([states, actions], dim=1)).size())

        return on_q_loss.item(), off_b_loss.item()

    def get_advantage(self, rewards, values, masks, gamma, tau):
        returns = torch.Tensor(rewards.size(0), 1).to(self.device)
        deltas = torch.Tensor(rewards.size(0), 1).to(self.device)
        advantages = torch.Tensor(rewards.size(0), 1).to(self.device)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + gamma * \
                prev_value * masks[i] - values.data[i]
            advantages[i] = deltas[i] + gamma * \
                tau * prev_advantage * masks[i]

            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]
            prev_advantage = advantages[i, 0]

        return returns, advantages

    def update_params(self, batch, off_buffers, num_horizons, i_epsiode):
        rewards = torch.Tensor(batch.reward).to(self.device)
        masks = torch.Tensor(batch.mask).to(self.device)
        actions = torch.Tensor(np.concatenate(batch.action, 0)).to(self.device)
        states = torch.Tensor(np.array(batch.state)).to(self.device)
        horizons = np.array(batch.horizon).flatten()
        with torch.no_grad():
            if FINITE:
                values = horizon_get_values(
                    states, horizons, self.policy_net, self.q_net, self.device)
            else:
                values = self.q_net.get_value(
                    Variable(states).to(self.device), self.policy_net)
        returns, advantages = self.get_advantage(
            rewards, values, masks, self.gamma, self.tau)

        # on batches contains the rewards, masks, actions, states, returns for each horizon
        on_batches = []
        for h in range(num_horizons):
            indexes = np.where(horizons == h)
            h_states = states[indexes, :].squeeze(0)
            h_rewards = rewards[indexes].reshape(-1, 1)
            h_returns = returns[indexes, :].squeeze(0)
            h_advantages = advantages[indexes, :].squeeze(0)
            h_masks = masks[indexes].reshape(-1, 1)
            h_actions = actions[indexes, :].squeeze(0)
            on_batches.append({'states': h_states, 'rewards': h_rewards,
                               'returns': h_returns, 'advantages': h_advantages, 'masks': h_masks, 'actions': h_actions})

        # print(len(rewards))
        # off_states, off_actions, off_rewards, off_next_states, off_not_dones, off_horizons = off_buffers.sample(
        #     np.minimum(off_buffers.capacity, len(rewards)))

        # on batches contains the states, actions, rewards, next states, not_dones for each horizon
        off_batches = []
        for h in range(num_horizons):
            off_batches.append(off_buffers[h].sample(
                np.minimum(off_buffers[h].capacity, len(rewards)//num_horizons)))

        on_losses = []
        off_losses = []
        # FQE
        for h in range(num_horizons - 1, -1, -1):
            off_states, off_actions, off_rewards, off_next_states, off_not_dones, _ = off_batches[
                h]
            off_rewards = off_rewards.to(self.device)
            with torch.no_grad():
                if h == num_horizons - 1:
                    off_targets = off_rewards
                else:
                    off_next_action_mean, _, off_next_action_std = self.policy_net[h+1](
                        off_next_states.to(self.device))
                    off_next_action = torch.normal(
                        off_next_action_mean, off_next_action_std)
                    off_targets = off_rewards + \
                        self.q_net[h+1](torch.cat([off_next_states,
                                        off_next_action], dim=1).to(self.device))

            on_loss, off_loss = self.update_q(
                on_batches[h]['states'], on_batches[h]['actions'], on_batches[h]['returns'],
                off_states, off_actions, off_targets, self.q_net[h])
            on_losses.append(on_loss)
            off_losses.append(off_loss)
        on_loss = np.mean(on_losses)
        off_loss = np.mean(off_losses)
        # on_loss, off_loss = self.update_q(
        #     states, actions, targets, off_states, off_actions, off_targets)
        # soft_update_params(self.q_net, self.target_q_net, self.critic_ratio)

        # update the advantages
        with torch.no_grad():
            if FINITE:
                values = horizon_get_values(
                    states, horizons, self.policy_net, self.q_net, self.device)
            else:
                values = self.q_net.get_value(
                    Variable(states).to(self.device), self.policy_net)
        returns, advantages = self.get_advantage(
            rewards, values, masks, self.gamma, self.tau)

        for h in range(num_horizons):
            indexes = np.where(horizons == h)
            h_advantages = advantages[indexes, :].squeeze(0)
            on_batches[h]['advantages'] = h_advantages

        off_advantages_list = []
        off_actions_list = []
        with torch.no_grad():
            for h in range(num_horizons):
                off_states, off_actions, _, _, _, _ = off_batches[
                    h]
                off_states = off_states.to(self.device)
                off_actions = off_actions.to(self.device)
                off_action_mean, _, off_action_std = self.policy_net[h](
                    off_states)
                off_actions = torch.normal(off_action_mean, off_action_std)
                off_states_actions = torch.cat(
                    [off_states, off_actions], dim=1)
                off_qs = self.q_net[h](off_states_actions)
                off_vs = self.q_net[h].get_value(
                    off_states, self.policy_net[h])
                off_advantages = off_qs - off_vs
                off_advantages_list.append(off_advantages)
                off_actions_list.append(off_actions)

        # for i, h in enumerate(off_horizons):
        #     print(h)
        #     print(off_states[i])
        #     assert off_states[i, h+3].item() == 1

        if FINITE:
            loss_befores = []
            loss_afters = []
            for h in range(num_horizons):
                states = on_batches[h]['states']
                actions = on_batches[h]['actions']
                advantages = on_batches[h]['advantages']
                off_states, _, _, _, _, _ = off_batches[
                    h]
                off_advantages = off_advantages_list[h]
                off_actions = off_actions_list[h]
                off_states = off_states.to(self.device)
                if HYBRID:
                    states = torch.cat([states, off_states], dim=0)
                    actions = torch.cat([actions, off_actions], dim=0)
                    advantages = torch.cat([advantages, off_advantages], dim=0)
                loss_before, loss_after = update_policy(
                    self.policy_net[h], states, actions, advantages, self.max_kl, self.damping, self.device)
                loss_afters.append(loss_after)
                loss_befores.append(loss_before)
            loss_before = np.mean(loss_befores)
            loss_after = np.mean(loss_afters)
        else:
            loss_before, loss_after = update_policy(
                self.policy_net, states, actions, advantages, self.max_kl, self.damping, self.device)
        return on_loss, off_loss, loss_before, loss_after

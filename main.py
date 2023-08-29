# import wandb
from itertools import count

# import gym
# import d4rl
import scipy.optimize
import os

import torch
from models import *
from replay_memory import Memory, ReplayBuffer
from running_state import ZFilter
from torch.autograd import Variable
from trpo import TRPO
from utils import *
from collect_offline import *

import typing

from lock_continuous import Lock, CifarLock


torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')


# env = gym.make(args.env_name)
# off_dataset = d4rl.qlearning_dataset(env)
# print(off_dataset.keys())
# not_dones = 1 - off_dataset['terminals']
# off_buffer = ReplayBuffer(off_dataset['observations'], off_dataset['actions'],
#                           off_dataset['rewards'], off_dataset['next_observations'], not_dones)
def evaluate(env, agent, num_episodes):
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        t = 0
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(obs, t)
            action = action.data[0].numpy()
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            t += 1
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards)


def main(args):
    set_seed_everywhere(args.seed)
    train_returns = deque(maxlen=50)
    test_returns = deque(maxlen=50)
    if args.env_name == 'lock':
        env = Lock()
        env.init(horizon=args.horizon,
             action_dim=10)
        eval_env = env
    elif args.env_name == 'cifarlock':
        eval_env = CifarLock()
        eval_env.init(horizon=args.horizon,
                    action_dim=10,
                    cifar_repo='/share/cuvl/yifei/hybrid_ppo/cifar-100-python/test')
    else:
        raise NotImplementedError
    eval_env.opt_a = env.opt_a
    eval_env.opt_b = env.opt_b
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    print(num_states)
    print(num_actions)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    off_buffers = collect_offline_buffer(256, env, 50000, device=device)
    agent = TRPO(num_states, num_actions, args.horizon,
                 args.l2_reg,
                 args.gamma,
                 args.tau,
                 args.max_kl,
                 args.damping,
                 args.critic_ratio,
                 device=device)

    env_steps = 0
    max_reach = 0
    for i_episode in range(1500):
        memory = Memory()

        num_steps = 0
        reward_batch = 0
        num_episodes = 0
        # print('checkpoint1')
        max_reach = 0
        while num_steps < args.batch_size:
            state = env.reset()
            #state = running_state(state)

            reward_sum = 0
            for t in range(10000):  # Don't infinite loop while learning
                action = agent.select_action(state, t)
                action = action.data[0].numpy()
                # print(action)
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward

                #next_state = running_state(next_state)

                mask = 1
                if done:
                    mask = 0

                memory.push(state, np.array(
                    [action]), mask, next_state, reward, t)

                if done:
                    max_reach = max(env.max_reach, max_reach)
                    break

                state = next_state
            num_steps += (t+1)
            num_episodes += 1
            reward_batch += reward_sum
        #     print(f'checkpoint2 in the loop, t={num_steps}')
        # print('checkpoint3 out of the loop')

        reward_batch /= num_episodes
        batch = memory.sample()
        on_loss, off_loss, loss_before, loss_after = agent.update_params(
            batch, off_buffers, args.horizon, i_episode)

        env_steps += num_steps
        test_reward_batch = evaluate(eval_env, agent, 30)
        train_returns.append(reward_batch)
        test_returns.append(test_reward_batch)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                i_episode, reward_sum, reward_batch))
        wandb.log({"env_steps": env_steps,
                   "train_episode_reward": np.mean(train_returns),
                   "test_episode_reward": np.mean(test_returns),
                   "on_loss": on_loss,
                   "off_loss": off_loss,
                   "loss_before": loss_before,
                   "loss_after": loss_after,
                   "max_reach": max_reach})


if __name__ == '__main__':

    args = parse_args()

    import wandb

    # comment this out if don't use wandb
    # os.environ['WANDB_MODE'] = 'offline'

    with wandb.init(
            project=f'lock-{args.horizon}-{args.seed}',
            job_type="ratio_search",
            config=vars(args),
            name="Finite-trpo"):
        main(args)
    # main(args)

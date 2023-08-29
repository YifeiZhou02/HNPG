import numpy as np
import gym
from gym.spaces import Discrete, Box
import scipy.linalg
import math
from PIL import Image
import random
# cifar lock involves precomputing image features
import torch
import torch.nn.functional as F
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
from transformers import CLIPModel
from transformers import ViTFeatureExtractor, ViTModel
import tqdm
'''
fast sampling. credit: https://stackoverflow.com/questions/34187130/fast-random-weighted-selection-across-all-rows-of-a-stochastic-matrix/34190035
'''


def sample(prob_matrix, items, n):

    cdf = np.cumsum(prob_matrix, axis=1)
    # random numbers are expensive, so we'll get all of them at once
    ridx = np.random.random(size=n)
    # the one loop we can't avoid, made as simple as possible
    idx = np.zeros(n, dtype=int)
    for i, r in enumerate(ridx):
        idx[i] = np.searchsorted(cdf[i], r)
    # fancy indexing all at once is faster than indexing in a loop
    return items[idx]


class Lock(gym.Env):
    """A (stochastic) combination lock environment.
    Can configure the length, dimension, and switching probability via env_config"""

    def __init__(self, env_config={}):
        self.initialized = False

    def init(self, horizon=100, action_dim=10, p_switch=0.5, p_anti_r=0.5, anti_r=0.1, noise=0.1, num_envs=10, temperature=0.1,
             variable_latent=False, dense=False):
        self.initialized = True
        self.max_reward = 1
        self.horizon = horizon
        self.state_dim = 3
        self.action_dim = action_dim
        self.action_space = Box(low=0.0, high=1.0, shape=(
            self.action_dim,), dtype=np.float)
        self._max_episode_steps = horizon

        self.reward_range = (0.0, 1.0)

        self.observation_dim = 2 ** int(math.ceil(np.log2(self.horizon+4)))

        self.observation_space = Box(low=0.0, high=1.0, shape=(
            self.observation_dim,), dtype=np.float)

        self.p_switch = p_switch
        self.p_anti_r = p_anti_r
        self.anti_r = anti_r
        self.noise = noise
        self.rotation = scipy.linalg.hadamard(self.observation_space.shape[0])

        self.num_envs = num_envs
        self.tau = temperature

        self.variable_latent = variable_latent
        self.dense = dense

        self.optimal_reward = 1
        if dense:
            self.step_reward = 0.1

        self.all_latents = np.arange(self.state_dim)

        self.opt_a = np.random.randint(
            low=0, high=self.action_dim, size=self.horizon)
        self.opt_b = np.random.randint(
            low=0, high=self.action_dim, size=self.horizon)

        print("[LOCK] Initializing Combination Lock Environment")
        print("[LOCK] A sequence: ", end="")
        print([z for z in self.opt_a])
        print("[LOCK] B sequence: ", end="")
        print([z for z in self.opt_b])

    def step(self, action):
        if self.h == self.horizon:
            raise Exception("[LOCK] Exceeded horizon")

        r = 0
        #rtmp = np.random.binomial(1,0.5)
        next_state = None
        ber = np.random.binomial(1, self.p_switch)
        ber_r = np.random.binomial(1, self.p_anti_r)
        action_exp = np.exp(action / self.tau)
        #print(action, )
        softmax = action_exp / action_exp.sum()
        action = np.random.choice(self.action_dim, 1, p=softmax)[0]
        #print(softmax, action, self.opt_a[self.h], self.opt_b[self.h])

        # First check for end of episode
        if self.h == self.horizon-1:
            # Done with episode, need to compute reward
            if self.state == 0 and action == self.opt_a[self.h]:
                r = 1
                next_state = 0
                self.max_reach = self.h
            elif self.state == 1 and action == self.opt_b[self.h]:
                r = 1
                next_state = 1
                self.max_reach = self.h
            else:
                if ber_r:
                    r = self.anti_r
                else:
                    r = 0
                next_state = 2
            self.h += 1
            self.state = next_state
            obs = self.make_obs(self.state)
            return obs, r, True, {}

        # Decode current state
        r = 0
        if self.state == 0:
            # In state A
            if action == self.opt_a[self.h]:
                if ber:
                    next_state = 1
                else:
                    next_state = 0
            else:
                self.max_reach = self.h
                if ber_r:
                    r = self.anti_r
                else:
                    r = 0
                next_state = 2
        elif self.state == 1:
            # In state B
            if action == self.opt_b[self.h]:
                if ber:
                    next_state = 0
                else:
                    next_state = 1
            else:
                self.max_reach = self.h
                if ber_r:
                    r = self.anti_r
                else:
                    r = 0
                next_state = 2
        else:
            # In state C
            next_state = 2
        self.h += 1
        self.state = next_state
        obs = self.make_obs(self.state)
        return obs, 0, False, {}

    def get_state(self):
        return self.state

    def get_counts(self):
        counts = np.zeros(3, dtype=np.int)
        # for i in range(self.num_envs):
        counts[self.state] += 1

        return counts

    def make_obs(self, s):

        gaussian = np.zeros(self.observation_space.shape)
        gaussian[:(self.horizon+self.state_dim)] = np.random.normal(0,
                                                                    self.noise, [self.horizon+self.state_dim])
        gaussian[s] += 1
        gaussian[self.state_dim+self.h] += 1

        x = (self.rotation*np.matrix(gaussian).T).T
        return np.reshape(np.array(x), x.shape[1])

    def sample_latent(self, obs):

        latent_exp = np.exp(self.latents / self.tau)

        softmax = latent_exp / latent_exp.sum(axis=-1, keepdims=True)
        self.state = sample(softmax, self.all_latents, self.num_envs)

    def generate_obs(self, s, h):

        gaussian = np.zeros((self.num_envs, self.observation_space.shape[0]))
        gaussian[:, :(self.horizon+self.state_dim)] = np.random.normal(0,
                                                                       self.noise, [self.num_envs, self.horizon+self.state_dim])
        gaussian[:, s] += 1
        gaussian[:, self.state_dim+h] += 1

        x = np.matmul(self.rotation, gaussian.T).T

        return x

    def trim_observation(self, o, h):
        return (o)

    def reset(self):
        if not self.initialized:
            raise Exception("Environment not initialized")
        self.h = 0
        self.max_reach = 0
        ber = np.random.binomial(1, self.p_switch)

        if ber:
            self.state = 0
        else:
            self.state = 1
        obs = self.make_obs(self.state)
        return (obs)

    def uniform_reset(self):
        if random.random() < 0.5:
            return self.reset()
        if not self.initialized:
            raise Exception("Environment not initialized")
        self.h = random.choice([i for i in range(self.horizon)])
        self.max_reach = self.h
        self.state = random.choice([0, 1, 2])
        obs = self.make_obs(self.state)
        return (obs)

    def render(self, mode='human'):
        if self.state == 0:
            print("A%d" % (self.h))
        if self.state == 1:
            print("B%d" % (self.h))
        if self.state == 2:
            print("C%d" % (self.h))


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073),
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.fromarray(c_data, 'RGB')
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


def extract_all_images(images, model, device, batch_size=64, num_workers=1, normalize=True):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            features = model.get_image_features(b)
            if normalize:
                features = F.normalize(features, p=2, dim=1)
            all_image_features.append(features)
    all_image_features = torch.cat(all_image_features, dim=0)
    return all_image_features


class CifarLock(Lock):
    """A (stochastic) combination lock environment.
    Can configure the length, dimension, and switching probability via env_config"""

    def __init__(self, env_config={}):
        self.initialized = False

    def init(self, horizon=100, action_dim=10, p_switch=0.5, p_anti_r=0.5, anti_r=0.1, noise=0.1, num_envs=10, temperature=0.1,
             variable_latent=False, dense=False, cifar_repo='/share/cuvl/yifei/hybrid_ppo/cifar-100-python/train'):
        super().init(horizon, action_dim, p_switch, p_anti_r,
                     anti_r, noise, num_envs, temperature, variable_latent, dense)
        self.observation_dim = 512

        self.observation_space = Box(
            low=0.0, high=1.0, shape=(self.observation_dim,), dtype=np.float)

        cifar_dict = unpickle(cifar_repo)
        filtered_images = [image.reshape(3, 32, 32).transpose(1, 2, 0)
                           for i, image in enumerate(list(cifar_dict[b'data'])) if cifar_dict[b'fine_labels'][i] < 3*self.horizon+3]
        filtered_labels = [
            label for label in cifar_dict[b'fine_labels'] if label < 3*self.horizon+3]

        clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32").cuda()
        # class_images = [image.reshape(3, 32, 32)/255.
        #                 for image in list(cifar_dict[b'data'])]
        image_embeddings = extract_all_images(
            filtered_images, clip, torch.device('cuda')).cpu().numpy()

        class_images = [[] for _ in range(100)]
        for image, label in zip(image_embeddings, filtered_labels):
            class_images[label].append(image)
        self.class_images = class_images

    def make_obs(self, s):
        return random.sample(self.class_images[3*self.h + s], 1)[0]

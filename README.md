# HNPG
This repository contains our hybrid NPG-style algorithm for the comblock environment.

## Prerequisites
Please see requirements.txt for required packages.

We use [wandb](https://wandb.ai/home) to perform result collection, please setup wandb before running the code or add `os.environ['WANDB_MODE'] = 'offline'` in `main.py`.

For Cifar100 Comblock, please download cifar-100 from [here](https://www.cs.toronto.edu/~kriz/cifar.html), and set the path in <code>main.py</code> to the path of your <code>cifar-100-python</code> folder.


## Offline Datasets
The offline dataset is collected by following $\pi^\star$ with $\epsilon$-greedy exploration with $\epsilon=1/H$.

Please refer to our paper for more details.

## Run our code

To reproduce our result in continuous comblock:
```bash
python main.py --seed 12345 --env-name lock --horizon 50
```
To reproduce our result in cifar100 continuous comblock:
```bash
python main.py --seed 12345 --env-name cifarlock --horizon 30
```
## Credit 
Some code are adapted from BRIEE https://github.com/yudasong/briee, and https://github.com/ikostrikov/pytorch-trpo.

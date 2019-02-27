# wurm

Vectorised implementation of the classic mobile game snake as a 
reinforcement learning environment.

As the environment is just a PyTorch Tensor and its `step()` method
is implemented with purely vectorised operations such as convolutions and
ReLu we can represent a batch of environments as a 4D tensor and evolve 
all of them in parallel. This very efficient - a 1080 Ti can run
\> 10^5 environments of size 12 in parallel leading to over a million
environment steps per second being processed.

_This repo is under active development._

### Setup

Create a Python 3.6 virtualenv and install from `requirements.txt`

### Usage

The following command will train a A3C agent. It should achieve an 
average length of 10 in around 20 million steps. This takes about two
minutes on a 1080 Ti.

```
python -m experiments.a2c --env snake --num-envs 1000 --size 9 --observation partial_2
```
import os

PATH = os.path.dirname(os.path.realpath(__file__))

DEFAULT_DEVICE = 'cuda:0'

FOOD_CHANNEL = 0
HEAD_CHANNEL = 1
BODY_CHANNEL = 2

EPS = 1e-6

# This probably won't change so I've just made it a hardcoded value
INPUT_CHANNELS = 3

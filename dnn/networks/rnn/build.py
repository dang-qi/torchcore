from ....util.registry import Registry
from ....util.build import build_with_config

RNN_REG = Registry('RNN')

def build_rnn(cfg):
    rnn = build_with_config(cfg, RNN_REG)
    return rnn

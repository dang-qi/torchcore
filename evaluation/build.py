from ..util.registry import Registry
from ..util.build import build_with_config

EVALUATOR_REG = Registry('Evaluator')

def build_evaluator(cfg):
    evaluator = build_with_config(cfg, EVALUATOR_REG)
    return evaluator

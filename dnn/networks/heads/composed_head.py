import torch.nn as nn

class ComposedHead(nn.Module):
    def __init__(self, heads_names, heads):
        super().__init__()
        assert len(heads) == len(heads_names)
        for head_name, head in zip(heads_names, heads):
            self.__setattr__(head_name, head)
        self.head_names = heads_names

    def forward(self, x):
        out = {}
        for name in self.head_names:
            out[name] = self.__getattr__(name)(x)
        return out



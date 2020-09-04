import torch.nn as nn

class ComposedListHead(nn.Module):
    def __init__(self, heads_names, heads):
        super().__init__()
        assert len(heads) == len(heads_names)
        self.head_len = {}
        for head_name, head_list in zip(heads_names, heads):
            self.__setattr__(head_name, head_list)
            self.head_len[head_name] = len(head_list)
        self.head_names = heads_names

    def forward(self, x_list):
        out = {}
        for name in self.head_names:
            out[name] = []
            for i in range(self.head_len[name]):
                out_temp = self.__getattribute__(name)[i](x_list[i])
                out[name].append(out_temp)
                
        return out



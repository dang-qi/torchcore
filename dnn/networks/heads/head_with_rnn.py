#from typing import OrderedDict
from collections import OrderedDict
import torch
from torch.nn.functional import interpolate
from torch import nn
from torch.nn import Module
from ..rnn import build_rnn
from .build import build_head

class FCOSHeadWithGrammarRNN(Module):
    def __init__(self, head_cfg, rnn_cfg, strides, ref_scale_ind, grammar):
        '''
            grammar(list((int, int, ...), (int, int, ..))): the grammar index, should start from 0
        '''
        super().__init__()
        self.head = build_head(head_cfg)
        self.rnn_cfg = rnn_cfg
        self.strides = strides
        self.ref_scale_ind = ref_scale_ind
        self.scales = [s/strides[ref_scale_ind] for s in strides]
        self.grammar = grammar
        self.build_rnn_by_grammar()
        self.grammar_map = self._gen_grammar_map()

    def forward(self, feature):
        # the output of head is per fpn layer output [(class, bbox, ceterness(optional)),...]
        head_out = self.head(feature)
        dict_keys = None
        if isinstance(head_out, OrderedDict):
            dict_keys = list(head_out.keys())
            head_out = list(head_out.values())
        elif isinstance(head_out, list):
            pass
        else:
            raise ValueError('do not support the input type {}'.format(type(head_out)))

        feature_class = [f[0] for f in head_out]

        feature_shapes = [f.shape for f in feature_class]
        out_shape = feature_shapes[self.ref_scale_ind]
        feature_class = [interpolate(f, size=out_shape[-2:], mode='bilinear',align_corners=True) for f in feature_class]
        feature_class = [f.unsqueeze(1) for f in feature_class]
        #for i,f in enumerate(feature_class):
        #    print(i,f.shape)

        # Nx5xCxHxW
        feature_class = torch.cat(feature_class, dim=1)
        # CxNx5xHxW
        feature_class = feature_class.permute(2,0,1,3,4)

        # select features for each grammar
        rnn_features = [torch.stack([feature_class[t] for t in g]) for g in self.grammar]

        rnn_out = []
        for i, rnn_feature in enumerate(rnn_features):
            out = getattr(self, 'rnn_{}'.format(i))(rnn_feature)
            rnn_out.append(out)
        #print('rnn out 0 mean', rnn_out[0].mean())
        
        # merge the features from each grammar by max pooling
        feature_class_out = self.merge_features_from_grammar(rnn_out, feature_class)

        # CxNx5xHxW to 5xNxCxHxW
        feature_class_out = feature_class_out.permute(2,1,0,3,4)

        # scale back
        feature_class_out = [interpolate(f, size=s[-2:],mode='bilinear',align_corners=True) for f,s in zip(feature_class_out, feature_shapes)]

        # convert the format of output
        head_out_new = []
        for h,f in zip(head_out,feature_class_out):
            #h[0]=f
            head_out_new.append((f, *h[1:]))
        if dict_keys is not None:
            head_out_new = OrderedDict([(k,v) for k,v in zip(dict_keys, head_out_new)])

        return head_out_new
    
    # merge the features by getting the maximum of each grid
    def merge_features_from_grammar(self, grammar_features, feature_class):
        feature_class = feature_class.clone()
        for class_ind, grammar_inds in self.grammar_map.items():
            if len(grammar_inds) == 1:
                feature_class[class_ind] = grammar_features[grammar_inds[0][0]][grammar_inds[0][1]]
            else:
                feature = grammar_features[grammar_inds[0][0]][grammar_inds[0][1]]
                for i,j in grammar_inds[1:]:
                    feature = torch.maximum(feature, grammar_features[i][j])
                feature_class[class_ind] = feature

        return feature_class 

    def _gen_grammar_map(self):
        grammar_map = {}
        # the j_th element from i_th grammar group
        for i,g in enumerate(self.grammar):
            for j,ind in enumerate(g):
                if ind not in grammar_map:
                    grammar_map[ind] = []
                grammar_map[ind].append((i,j))
        return grammar_map


    def build_rnn_by_grammar(self):
        for i,g in enumerate(self.grammar):
            setattr(self, 'rnn_{}'.format(i), build_rnn(self.rnn_cfg))

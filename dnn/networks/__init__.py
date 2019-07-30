from easydict import EasyDict as edict
from . import feature

networks = edict()
networks.feature = {}
#networks.feature.vgg_small = vgg.vgg_small
networks.feature['vgg_small'] = feature.VggSmall
networks.feature['vgg16'] = feature.Vgg16
networks.feature['EasyNetA'] = feature.EasyNetA
networks.feature['EasyNetB'] = feature.EasyNetB
networks.feature['EasyNetC'] = feature.EasyNetC
networks.feature['EasyNetD'] = feature.EasyNetD

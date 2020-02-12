from easydict import EasyDict as edict
from . import feature
from . import necks

networks = edict()
networks.feature = {}
#networks.feature.vgg_small = vgg.vgg_small
networks.feature['vgg_small'] = feature.VggSmall
networks.feature['vgg16'] = feature.Vgg16
networks.feature['EasyNetA'] = feature.EasyNetA
networks.feature['EasyNetB'] = feature.EasyNetB
networks.feature['EasyNetC'] = feature.EasyNetC
networks.feature['EasyNetD'] = feature.EasyNetD
networks.feature['resnet18'] = feature.resnet18
networks.feature['resnet34'] = feature.resnet34
networks.feature['resnet50'] = feature.resnet50
networks.feature['resnet101'] = feature.resnet101
networks.feature['resnet152'] = feature.resnet152

neck = {}
neck['upsample_basic'] = necks.upsample_basic_3
networks.neck = neck


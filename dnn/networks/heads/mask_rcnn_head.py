from torch import nn

from .build import HEAD_REG

@HEAD_REG.register()
class MaskRCNNHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.class_num = cfg.class_num
        in_channel = cfg.in_channel
        self.feature_head = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=256,kernel_size=3,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,kernel_size=3,stride=1),
            nn.ReLU(inplace=True),
        )
        self.pred_head = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=self.class_num,kernel_size=1, stride=1, padding=0)
        )

        self.inin_weight()

    def inin_weight(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forword(self, feature):
        feature = self.feature_head(feature)
        result = self.pred_head(feature)
        return result

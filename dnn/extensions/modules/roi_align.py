import torch
import torch.nn as nn

from .crop_and_resize import CropAndResizeFunction

class RoiAlign(nn.Module):
    def __init__( self, crop_height, crop_width, extrapolation_value=0, transform_fpcoor=True ):
        super().__init__()

        self._crop_height = crop_height
        self._crop_width = crop_width
        self._extrapolation_value = extrapolation_value
        self._transform_fpcoor = transform_fpcoor

    def forward( self, featuremap, boxes, box_ind ):
        """
        RoIAlign based on crop_and_resize.
        :param featuremap: NxCxHxW
        :param boxes: Mx4 float box with (x1, y1, x2, y2) **without normalization**
        :param box_ind: M
        :return: MxCxoHxoW
        """

        x1, y1, x2, y2 = torch.split(boxes, 1, dim=1)
        image_height, image_width = featuremap.size()[2:4]
        if self._transform_fpcoor:
            spacing_w = (x2 - x1) / float(self._crop_width)
            spacing_h = (y2 - y1) / float(self._crop_height)

            nx0 = (x1 + spacing_w / 2 - 0.5) / float(image_width - 1)
            ny0 = (y1 + spacing_h / 2 - 0.5) / float(image_height - 1)
            nw = spacing_w * float(self._crop_width - 1) / float(image_width - 1)
            nh = spacing_h * float(self._crop_height - 1) / float(image_height - 1)

            boxes = torch.cat((ny0, nx0, ny0 + nh, nx0 + nw), 1)
        else:
            x1 = x1 / float(image_width - 1)
            x2 = x2 / float(image_width - 1)
            y1 = y1 / float(image_height - 1)
            y2 = y2 / float(image_height - 1)
            boxes = torch.cat((y1, x1, y2, x2), 1)

        return CropAndResizeFunction(self._crop_height, self._crop_width,
                                     self._extrapolation_value)(featuremap, boxes, box_ind)

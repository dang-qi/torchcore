import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import roipool_cpu
if torch.cuda.is_available() :
    import roipool_gpu

class RoiPoolFunction(Function):
    def __init__( self, crop_height, crop_width, spatial_scale ):
        self._crop_height = crop_height
        self._crop_width = crop_width
        self._spatial_scale = spatial_scale

    def forward( self, data, rois, roibatches ):
        assert( data.dtype == torch.float32 )
        assert( rois.dtype == torch.float32 )
        assert( roibatches.dtype == torch.int32 )

        device = data.device
        nrois = rois.size()[0]
        channels = data.size()[1]

        crops = torch.zeros( [ nrois, channels, self._crop_height, self._crop_height ],
                             dtype=data.dtype, device=device )
        argmaxes = torch.zeros( [nrois, channels, self._crop_height, self._crop_width ],
                             dtype=torch.int32, device=device )

        if data.is_cuda :
            roipool_gpu.forward( data, rois, roibatches, self._spatial_scale, crops, argmaxes )
        else :
            roipool_cpu.forward( data, rois, roibatches, self._spatial_scale, crops, argmaxes )

        self.data_size = data.size()
        self.save_for_backward( argmaxes, rois, roibatches )

        return crops.to( device )

    def backward( self, crops_grad ):
        assert( crops_grad.dtype == torch.float32 )
        device = crops_grad.device

        data_size = self.data_size
        argmaxes, rois, roibatches = self.saved_tensors

        crops_grad = crops_grad.contiguous()
        data_grad = torch.zeros( data_size, dtype=crops_grad.dtype,
                                            device=device )

        if crops_grad.is_cuda :
            roipool_gpu.backward( crops_grad, argmaxes, rois, roibatches,
                                self._spatial_scale, data_grad )
        else :
            roipool_cpu.backward( crops_grad, argmaxes, rois, roibatches,
                                self._spatial_scale, data_grad )

        return data_grad.to(device), None, None

class RoiPool(nn.Module):
    def __init__( self, crop_height, crop_width ):
        super().__init__()
        self._crop_height = crop_height
        self._crop_width = crop_width

    def forward( self, data, rois, roibatches, spatial_scale ):
        return RoiPoolFunction( self._crop_height, self._crop_width, spatial_scale )( data, rois, roibatches )

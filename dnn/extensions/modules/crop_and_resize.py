import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

#import _init
import crop_and_resize_cpu
if torch.cuda.is_available() :
    import crop_and_resize_gpu

class CropAndResizeFunction(Function):
    def __init__( self, crop_height, crop_width, extrapolation_value=0 ):
        super().__init__()
        self._crop_height = crop_height
        self._crop_width = crop_width
        self._extrapolation_value = extrapolation_value

    def forward( self, image, boxes, box_indices ):
        assert( image.dtype == torch.float32 )
        assert( boxes.dtype == torch.float32 )
        assert( box_indices.dtype == torch.int32 )

        device = image.device

        image = image.cpu()
        boxes = boxes.cpu()
        box_indices = box_indices.cpu()


        crops = torch.zeros_like( image ).resize_( boxes.size()[0],
                                                   image.size()[1],
                                                   self._crop_height,
                                                   self._crop_width )

        #if image.is_cuda :
        #    crop_and_resize_gpu.forward( image, boxes, box_indices,
        #                                             self._extrapolation_value, self._crop_height,
        #                                             self._crop_width,
        #                                             crops )
        #else :
        crop_and_resize_cpu.forward( image, boxes, box_indices,
                                    self._extrapolation_value, self._crop_height,
                                    self._crop_width, crops )
        self.im_size = image.size()
        self.save_for_backward( boxes, box_indices )
        return crops.to( device )

    def backward( self, grad_outputs ):
        assert( grad_outputs.dtype == torch.float32 )
        device = grad_outputs.device

        grad_outputs = grad_outputs.cpu()
        boxes, box_indices = self.saved_tensors

        grad_outputs = grad_outputs.contiguous()
        image_grad = torch.zeros_like(grad_outputs).resize_(*self.im_size)

        #if grad_outputs.is_cuda :
        #    crop_and_resize_gpu.backward( grad_outputs, boxes, box_indices, image_grad )
        #else :
        crop_and_resize_cpu.backward( grad_outputs, boxes, box_indices, image_grad )

        return image_grad.to(device), None, None

class CropAndResize(nn.Module):
    def __init__( self, crop_height, crop_width, extrapolation_value=0 ):
        super().__init__()
        self._crop_height = crop_height
        self._crop_width = crop_width
        self._extrapolation_value = extrapolation_value

    def forward( self, image, boxes, box_indices ):
        return CropAndResizeFunction( self._crop_height, self._crop_width,
                                      self._extrapolation_value )( image, boxes, box_indices )

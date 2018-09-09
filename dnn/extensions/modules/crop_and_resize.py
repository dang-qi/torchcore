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
		device = image.device
		crops = torch.zeros( [ boxes.size()[0], image.size()[1], self._crop_height, self._crop_width ],
							  dtype=torch.float32, device=device )

		if image.is_cuda :
			crop_and_resize_gpu.forward( image, boxes, box_indices,
                                                     self._extrapolation_value, self._crop_height,
                                                     self._crop_width,
                                                     crops )
		else :
			crop_and_resize_cpu.forward( image, boxes, box_indices,
			                             self._extrapolation_value, self._crop_height, 
                                                     self._crop_width,
						     crops )
		self.im_size = image.size()
		self.save_for_backward( boxes, box_indices )
		return crops

	def backward( self, grad_outputs ):
		boxes, box_indices = self.saved_tensors
		device = grad_ouputs.device
		image_grad = torch.zeros( self.im_size, dtype=torch.float32, device=device )

		if grad_ouputs.is_cuda :
			crop_and_resize_gpu.backward( grad_ouputs, boxes, box_indices, image_grad )
		else :
			crop_and_resize_cpu.backward( grad_ouputs, boxes, box_indices, image_grad )

		return image_grad, None, None

class CropAndResize(nn.Module):
	def __init__( self, crop_height, crop_width, extrapolation_value=0 ):
		super().__init__()
		self._crop_height = crop_height
		self._crop_width = crop_width
		self._extrapolation_value = extrapolation_value

	def forward( self, image, boxes, box_indices ):
		return CropAndResizeFunction( self._crop_height,
									  self._crop_width,
									  self._extrapolation_value )( image, boxes, box_indices )

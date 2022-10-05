from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from .build import TRANSFORM_REG

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

@TRANSFORM_REG.register()
class CLIPImageTransform():
    def __init__(self, resolution) -> None:
        self.resolution = resolution
        self._transform = _transform(resolution)

    def __call__(self, inputs, targets):
        inputs['data'] = self._transform(inputs['data'])
        return inputs, targets
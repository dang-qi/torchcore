import torch
def set_device( blobs, device ):
    if type(blobs) == list:
        for i in range(len(blobs)):
            blobs[i] = set_device(blobs[i], device)
    elif type(blobs) == dict:
        for key, data in blobs.items():
            blobs[key] = set_device(data, device)
    elif torch.is_tensor(blobs):
        blobs = blobs.to(device)
    return blobs

class ImageExtractor():
    def __init__(self, dataset, id_key='image_id') -> None:
        self.dataset = dataset
        self.id_key = id_key
        self.index_dict = dict()
        self.build_index_dict()

    def build_index_dict(self):
        for i,im in enumerate(self.dataset):
            inputs, targets = im
            im_id = targets[self.id_key]
            self.index_dict[im_id] = i

    def extract(self, image_id):
        index = self.index_dict[image_id]
        return self.dataset[index]

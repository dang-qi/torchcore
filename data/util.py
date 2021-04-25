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
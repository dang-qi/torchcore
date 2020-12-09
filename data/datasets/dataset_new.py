import os
class Dataset:
    '''Basic Dataset'''
    _repr_indent = 4
    def __init__(self, root, transforms=None ):
        root = os.path.expanduser(root)
        self._root = root
        self._transforms=transforms

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of data: {}".format(self.__len__())]
        if self._root is not None:
            body.append("Root location: {}".format(self._root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self._transforms is not None:
            body += [repr(self._transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def extra_repr(self):
        return ""

    def convert_to_xyxy(self):
        for image in self._images:
            for obj in image['objects']:
                obj['bbox'][2]+=obj['bbox'][0]
                obj['bbox'][3]+=obj['bbox'][1]
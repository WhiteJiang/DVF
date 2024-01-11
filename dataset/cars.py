from .base import *
import shutil

import scipy.io
import os


class Cars(BaseDatasetMod):
    def __init__(self, root, source, classes, transform=None, image_dir=None):
        BaseDatasetMod.__init__(self, root, source, classes, transform)
        index = 0
        path_root = os.path.join(root, image_dir)
        for i in torchvision.datasets.ImageFolder(root=path_root).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes:
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(os.path.join(root, i[0]))
                index += 1


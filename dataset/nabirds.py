# -*- coding: utf-8 -*-
# @Time    : 2023/11/16
# @Author  : White Jiang
from .base import *
import torch


class Nabirds(BaseDatasetMod):
    def __init__(self, root, source, classes, transform=None):
        BaseDatasetMod.__init__(self, root, source, classes, transform)
        index = 0
        for i in torchvision.datasets.ImageFolder(root=
                                                  os.path.join(root, 'images')).imgs:
            y = i[1]  # label
            fn = os.path.split(i[0])[1]  # image name
            if y in self.classes:
                self.ys += [y]
                self.I += [index]
                self.im_paths.append(os.path.join(root, i[0]))
                index += 1

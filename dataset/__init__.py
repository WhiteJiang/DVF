from .cars import Cars
from .cub import CUBirds
from .nabirds import Nabirds
from . import utils

# import resource
#
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# print(rlimit)
# resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))

_type = {
    'cars': Cars,
    'nabirds': Nabirds,
    'cub': CUBirds,
}


def load(name, root, source, classes, transform=None):
    return _type[name](root=root, source=source, classes=classes, transform=transform)


def load_inshop(name, root, source, classes, transform=None, dset_type='train'):
    return _type[name](root=root, source=source, classes=classes, transform=transform, dset_type=dset_type)

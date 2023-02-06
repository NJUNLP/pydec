from .meta import CopyModule
from .module import DecModule
import torch.nn as nn


class Container(nn.Container, DecModule, metaclass=CopyModule):
    __doc__ = nn.Container.__doc__


class Sequential(nn.Sequential, DecModule, metaclass=CopyModule):
    __doc__ = nn.Sequential.__doc__


class ModuleList(nn.ModuleList, DecModule, metaclass=CopyModule):
    __doc__ = nn.ModuleList.__doc__


class ModuleDict(nn.ModuleDict, DecModule, metaclass=CopyModule):
    __doc__ = nn.ModuleDict.__doc__


class ParameterList(nn.ParameterList, DecModule, metaclass=CopyModule):
    __doc__ = nn.ParameterList.__doc__


class ParameterDict(nn.ParameterDict, DecModule, metaclass=CopyModule):
    __doc__ = nn.ParameterDict.__doc__

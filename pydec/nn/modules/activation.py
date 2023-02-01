import torch.nn as nn
from pydec import Composition
from .. import functional as F
from .module import DecModule
from .meta import ProxyModule



class ReLU(nn.ReLU, DecModule, metaclass=ProxyModule):
    def pydec_forward(self, input: Composition) -> Composition:
            return F.relu(input, inplace=self.inplace)
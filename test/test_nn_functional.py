import pydec
from pydec import Composition
import torch


def init_composition(size, c_num=3, requires_grad=False):
    c = Composition(size, c_num, dtype=torch.float)
    for i in range(c_num):
        c[i] = torch.randn(size, requires_grad=requires_grad)
    return c


class TestRelu:
    def test1(self):
        c = init_composition((3, 4))
        x = c.c_sum()
        x = torch.nn.functional.relu(x)
        c = pydec.nn.functional.relu(c)
        assert c.numc() == 3
        pydec.check_error(c, x)

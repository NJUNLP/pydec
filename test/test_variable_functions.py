import pydec
from pydec import Composition
import torch
import pytest
from .test_composition import init_composition

torch.manual_seed(114514)


@pytest.mark.filterwarnings("ignore:An output with one or more elements was resized")
class TestCat:
    c_list = [init_composition((i + 1, 2)) for i in range(3)]

    def test1(self):
        out = pydec.cat(self.c_list, dim=0)
        assert out.c_size() == (3, 6, 2)

    def test2(self):
        out = init_composition((6, 2, 1))
        out1 = pydec.cat(self.c_list, dim=0, out=out)
        assert out.c_size() == (3, 6, 2)
        assert (out1 == out).all()

    def test3(self):
        with pytest.raises(RuntimeError):
            pydec.cat(tuple(self.c_list), dim=1)


class TestCCat:
    c_list = [init_composition((3, 2), i + 1) for i in range(3)]

    def test1(self):
        out = pydec.c_cat(self.c_list)
        assert out.c_size() == (6, 3, 2)

    def test2(self):
        out = init_composition((3, 2), 6)
        out1 = pydec.c_cat(self.c_list, out=out)
        assert out.c_size() == (6, 3, 2)
        assert (out1._composition_tensor == out._composition_tensor).all()

    def test3(self):
        with pytest.raises(RuntimeError):
            pydec.c_cat(self.c_list + [init_composition((2, 2))])


class TestApply:
    def test_abs(self):
        c = init_composition((3, 4))
        out = pydec.c_apply(c, torch.abs)
        assert (out._composition_tensor == c._composition_tensor.abs()).all()


class TestDiagonalInit:
    c = init_composition((2, 3))

    def test1(self):
        data = torch.zeros((2, 3))

        out = pydec.diagonal_init(self.c, data, 1)

        for i in range(self.c.numc()):
            for j in range(self.c.numc()):
                if i == j:
                    assert torch.all(out[i, :, j] == data[:, i])
                else:
                    assert torch.all(out[i, :, j] == self.c[i, :, j])

    def test2(self):
        c = init_composition((2, 3), 4)

        data = torch.zeros((2, 3))
        offset = -1

        out = pydec.diagonal_init(c, data, 1, offset=offset)

        for i in range(1, c.numc()):
            for j in range(c.size(1)):
                if i + offset == j:
                    assert torch.all(out[i, :, j] == data[:, j])
                else:
                    assert torch.all(out[i, :, j] == c[i, :, j])

    def test3(self):
        component_num = 3
        size = (3, 2)
        x = torch.randn(size)
        c = pydec.zeros((3, 2), component_num)
        out = pydec.diagonal_init(c, src=x, dim=0)
        for i in range(self.c.numc()):
            for j in range(self.c.numc()):
                if i == j:
                    assert torch.all(out[i, j] == x[i])
                else:
                    assert torch.all(out[i, j] == c[i, j])

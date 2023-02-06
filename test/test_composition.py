import pydec
from pydec import Composition
import torch
import pytest

torch.manual_seed(114514)


def init_composition(size, c_num=3, requires_grad=False):
    c = pydec.zeros(size, c_num, dtype=torch.float)
    for i in range(c_num):
        c[i] = torch.randn(size, requires_grad=requires_grad)
    return c


class TestIndexing:
    @classmethod
    def init_composition(cls) -> Composition:
        size = (3, 4)
        c = pydec.zeros(size, c_num=4)
        return c

    def test1(self):
        c = TestIndexing.init_composition()
        with pytest.raises(RuntimeError):
            c[None]
        with pytest.raises(RuntimeError):
            c[None] = 1

    def test2(self):
        c = TestIndexing.init_composition()
        c0 = c[0]
        assert isinstance(c0, torch.Tensor)
        assert c0.size() == c.size()
        c02 = c[0:2]
        assert isinstance(c02, Composition)
        assert c02.numc() == 2
        assert c02.size() == c.size()
        c02_ = c[:, 0:2]
        assert c02_.numc() == c.numc()
        assert c02_.size() == (2,) + c.size()[1:]
        c0202 = c[:2, :2]
        assert c0202.numc() == 2
        assert c0202.size() == (2,) + c.size()[1:]

    def test3(self):
        c = TestIndexing.init_composition()
        c_ = c[:, None]
        assert c_.size() == (1,) + c.size()
        index_list = [0, 2]
        c_ = c[index_list]
        assert isinstance(c_, Composition)
        assert c_.numc() == 2
        assert c_.size() == c.size()
        c_ = c[index_list, index_list]
        assert c_.c_size() == (2,) + c.c_size()[2:]

        c_ = c[index_list, index_list, index_list]
        assert c_.c_size() == (2,)

        c_ = c[:, index_list, index_list]
        assert c_.c_size() == (4, 2)


class TestIndexingInAutotracing:
    @classmethod
    def init_composition(cls) -> Composition:
        size = (3, 4)
        c = Composition(size, component_num=4)
        return c

    def test1(self):
        c = TestIndexing.init_composition()
        with pydec.autotracing.set_tracing_enabled(True):
            c_ = c[None]
            assert c_.numc() == c.numc()
            assert c_.size() == (1,) + c.size()

            c[None] = 1
            assert torch.all(c._residual_tensor == 0)
            assert torch.all(c._composition_tensor == 1)

    def test2(self):
        c = TestIndexing.init_composition()
        with pydec.autotracing.set_tracing_enabled(True):
            c0 = c[0]
            assert isinstance(c0, Composition)
            assert c0.numc() == c.numc()
            assert c0.size() == c.size()[1:]
            c02 = c[0:2]
            assert isinstance(c02, Composition)
            assert c02.numc() == c.numc()
            assert c02.size() == (2,) + c.size()[1:]
            c02_ = c[:, 0:2]
            assert c02_.numc() == c.numc()
            assert c02_.size() == c.size()[:1] + (2,)
            c0202 = c[:2, :2]
            assert c0202.numc() == c.numc()
            assert c0202.size() == (2, 2)

    def test3(self):
        c = TestIndexing.init_composition()
        with pydec.autotracing.set_tracing_enabled(True):
            c_ = c[:, None]
            assert c_.size() == c.size()[:1] + (1,) + c.size()[1:]
            index_list = [0, 2]
            c_ = c[index_list]
            assert isinstance(c_, Composition)
            assert c_.numc() == c.numc()
            assert c_.size() == (2,) + c.size()[1:]
            c_ = c[index_list, index_list]
            assert c_.c_size() == (c.numc(), 2)

            with pytest.raises(IndexError):
                c_ = c[index_list, index_list, index_list]

            with pytest.raises(IndexError):
                c_ = c[:, index_list, index_list]


class TestView:
    c = init_composition((2, 3))

    def test_view1(self):
        assert self.c.view(torch.float16)._composition_tensor.dtype == torch.float16

        assert (
            self.c.view(dtype=torch.float16)._composition_tensor.dtype == torch.float16
        )

    def test_view2(self):
        assert self.c.view((3, 2)).size() == (3, 2)
        assert self.c.view(size=(3, 2)).size() == (3, 2)
        assert self.c.view(3, 2).size() == (3, 2)


class TestReshape:
    c = init_composition((2, 3))

    def test1(self):
        assert self.c.reshape((3, 2)).size() == (3, 2)
        assert self.c.reshape(shape=(3, 2)).size() == (3, 2)
        assert self.c.reshape(3, 2).size() == (3, 2)


class TestPlus:
    c = init_composition((2, 3))

    def test1(self):
        from pydec import set_decomposition_func

        self.c[:] = 1.5
        set_decomposition_func("abs_decomposition")
        c = self.c + 3
        assert (self.c._residual_tensor + 3 == c._residual_tensor).all()

        # set_decomposition_func("norm_decomposition")
        # c = self.c + 3
        # assert (self.c._residual_tensor + 3 == c._residual_tensor).all()

        set_decomposition_func("hybrid_decomposition")
        c = self.c + 3
        assert (self.c._residual_tensor + 3 == c._residual_tensor).all()

        # set_decomposition_func("sign_decomposition")
        # c = self.c + 3
        # assert (self.c._residual_tensor + 3 == c._residual_tensor).all()

        # set_decomposition_func("hybrid_decomposition_threshold")
        # c = self.c + 3
        # assert (self.c._residual_tensor + 3 == c._residual_tensor).all()

        with pydec.no_decomposition():
            c = self.c + 3
            assert (self.c._composition_tensor == c._composition_tensor).all()
            assert (self.c._residual_tensor + 3 == c._residual_tensor).all()

    def test2(self):
        c = init_composition((2, 3))
        out = c + self.c
        assert torch.all(
            self.c._composition_tensor + c._composition_tensor
            == out._composition_tensor
        )

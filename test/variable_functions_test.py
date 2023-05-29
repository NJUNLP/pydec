import pydec
from pydec import Composition
import torch
import pytest
from ._composition_test import init_composition

torch.manual_seed(114514)


class TestCat:
    c_list = [init_composition((i + 1, 2)) for i in range(3)]

    def test1(self):
        out = pydec.cat(self.c_list, dim=0)
        assert out.c_size() == (3, 6, 2)

    def test2(self):
        out_holder = init_composition((6, 2))
        out = pydec.cat(self.c_list, dim=0, out=out_holder)
        assert out_holder.c_size() == (3, 6, 2)
        assert (out == out_holder).all()

    def test3(self):
        with pytest.raises(RuntimeError):
            pydec.cat(tuple(self.c_list), dim=1)


class TestCCat:
    c_list = [init_composition((3, 2), i + 1) for i in range(3)]

    def test1(self):
        out = pydec.c_cat(self.c_list)
        assert out.c_size() == (6, 3, 2)

    def test2(self):
        out_holder = init_composition((3, 2), 6)
        out = pydec.c_cat(self.c_list, out=out_holder)
        assert out_holder.c_size() == (6, 3, 2)
        assert (out._component_tensor == out_holder._component_tensor).all()

    def test3(self):
        with pytest.raises(RuntimeError):
            pydec.c_cat(self.c_list + [init_composition((2, 2))])


class TestStack:
    c_list = [init_composition((2, 2)) for i in range(3)]

    def test1(self):
        out = torch.stack(self.c_list, dim=0)
        assert out.c_size() == (3, 3, 2, 2)

    def test2(self):
        out_holder = init_composition((3, 2, 2))
        out = torch.stack(self.c_list, dim=0, out=out_holder)
        assert out_holder.c_size() == (3, 3, 2, 2)
        assert (out == out_holder).all()

    def test3(self):
        with pytest.raises(RuntimeError):
            torch.stack(self.c_list + [init_composition((3, 2))], dim=0)


class TestCStack:
    t_list = [torch.rand((2, 2)) for i in range(3)]

    def test1(self):
        out = pydec.c_stack(self.t_list)
        assert out.c_size() == (3, 2, 2)

    def test2(self):
        out_holder = pydec.zeros((2, 2), 3)
        out = pydec.c_stack(self.t_list, out=out_holder)
        assert out_holder.c_size() == (3, 2, 2)
        assert (out._component_tensor == out_holder._component_tensor).all()

    def test3(self):
        with pytest.raises(RuntimeError):
            pydec.c_cat(self.t_list + [torch.rand((3, 2))])


class TestDiagonalInit:
    c = init_composition((2, 3))

    def test1(self):
        data = torch.zeros((2, 3))

        out = pydec.diagonal_init(self.c, data, 1)

        for i in range(self.c.numc()):
            for j in range(self.c.numc()):
                if i == j:
                    assert torch.all(out()[i, :, j] == data[:, i])
                else:
                    assert torch.all(out()[i, :, j] == self.c()[i, :, j])

    def test2(self):
        c = init_composition((2, 3), 4)

        data = torch.zeros((2, 3))
        offset = -1

        out = pydec.diagonal_init(c, data, 1, offset=offset)

        for i in range(1, c.numc()):
            for j in range(c.size(1)):
                if i + offset == j:
                    assert torch.all(out()[i, :, j] == data[:, j])
                else:
                    assert torch.all(out()[i, :, j] == c()[i, :, j])

    def test3(self):
        component_num = 3
        size = (3, 2)
        x = torch.randn(size)
        c = pydec.zeros((3, 2), component_num)
        out = pydec.diagonal_init(c, src=x, dim=0)
        for i in range(self.c.numc()):
            for j in range(self.c.numc()):
                if i == j:
                    assert torch.all(out()[i, j] == x[i])
                else:
                    assert torch.all(out()[i, j] == c()[i, j])


class TestCApply:
    def test_abs(self):
        c = init_composition((3, 4))
        out = pydec.c_apply(c, torch.abs)
        assert (out._component_tensor == c._component_tensor.abs()).all()


class TestCMap:
    def test_add(self):
        c1 = init_composition((3, 4))
        c2 = init_composition((4,))
        out = pydec.c_map(c1, c2, torch.add)
        assert (
            out._component_tensor == c1._component_tensor + c2._component_tensor
        ).all()


class TestNumel:
    def test1(self):
        c1 = init_composition((3, 4))
        assert c1.numel() == 3 * 4


class TestCNumel:
    def test1(self):
        c1 = init_composition((3, 4))
        assert c1.c_numel() == 3 * 4 * c1.numc()


class TestNumc:
    def test1(self):
        c1 = init_composition((3, 4), 10)
        assert c1.numc() == 10


class TestClone:
    def test1(self):
        c1 = init_composition((3, 4), 10)
        c1[:] = 1.1
        c1.residual[:] = 3.5
        c2 = torch.clone(c1)
        assert (c1.components == c2.components).all()
        assert (c1.residual == c2.residual).all()
        c1[:] = 2.0
        c1.residual[:] = 4.0
        assert not (c1.components == c2.components).any()
        assert not (c1.residual == c2.residual).any()


class TestDetach:
    def test1(self):
        size = (3, 4)
        numc = 10
        c1 = init_composition(size, numc)
        leaf_tensors = []
        for i in range(numc):
            leaf_tensors.append(torch.randn(size, requires_grad=True))
            c1()[i] = leaf_tensors[i]
        loss = (c1 * 2 + torch.detach(c1 * 3)).c_sum().sum()
        loss.backward()
        print(leaf_tensors[0].grad)
        assert (leaf_tensors[0].grad == 2).all()

    def test2(self):
        size = (3, 4)
        numc = 10
        c1 = init_composition(size, numc)
        leaf_tensors = []
        for i in range(numc):
            leaf_tensors.append(torch.randn(size, requires_grad=True))
            c1()[i] = leaf_tensors[i]
        loss = (c1 * 2 + torch.detach_(c1 * 3)).c_sum().sum()
        loss.backward()
        print(leaf_tensors[0].grad)
        assert (leaf_tensors[0].grad == 2).all()


class TestAdd:
    def test1(self):
        c1 = init_composition((3, 4))
        c2 = init_composition((4,))
        out = torch.add(c1, c2)
        assert (out.components == c1.components + c2.components).all()

    def test2(self):
        c1 = init_composition((3, 4))
        c2 = init_composition((4,))
        out = torch.add(c1, c2, alpha=2)
        assert (out.components == c1.components + 2 * c2.components).all()

    def test3(self):
        c1 = init_composition((3, 4))
        t1 = torch.rand((4,))
        out = torch.add(c1, t1)
        assert (out.components == c1.components).all()
        assert (out.residual == t1).all()

    def test4(self):
        c1 = init_composition((3, 4))
        c2 = init_composition((4,))
        out_holder = pydec.zeros((3, 4), 3)
        out = torch.add(c1, c2, out=out_holder)
        assert (out._component_tensor == out_holder._component_tensor).all()


class TestVar:
    def test1(self):
        c_input = init_composition((2, 3, 4))
        input = c_input.c_sum()

        ref = torch.var(input)
        c_out = torch.var(c_input)
        assert (ref - c_out.c_sum()).abs().sum() < 1e-3

        ref = torch.var(input, False)
        c_out = torch.var(c_input, False)
        assert (ref - c_out.c_sum()).abs().sum() < 1e-3

    def test2(self):
        c_input = init_composition((2, 3, 4))
        input = c_input.c_sum()
        ref = torch.var(input, -1)
        c_out = torch.var(c_input, -1)
        assert (ref - c_out.c_sum()).abs().sum() < 1e-3

        ref = torch.var(input, -1, unbiased=False)
        c_out = torch.var(c_input, -1, unbiased=False)
        assert (ref - c_out.c_sum()).abs().sum() < 1e-3

    def test3(self):
        c_input = init_composition((2, 3, 4))
        input = c_input.c_sum()
        ref = torch.var(input, dim=(0, 2))
        c_out = torch.var(c_input, dim=(0, 2))
        assert (ref - c_out.c_sum()).abs().sum() < 1e-3

        ref = torch.var(input, dim=(0, 2), unbiased=False)
        c_out = torch.var(c_input, dim=(0, 2), unbiased=False)
        assert (ref - c_out.c_sum()).abs().sum() < 1e-3


class TestBmm:
    def test1(self):
        c_input1 = init_composition((2, 3, 4))
        c_input2 = init_composition((2, 4, 5))
        input1 = c_input1.c_sum()
        input2 = c_input2.c_sum()
        ref = torch.bmm(input1, input2)
        c_out = torch.bmm(c_input1, c_input2)
        assert (ref - c_out.c_sum()).abs().sum() < 1e-3

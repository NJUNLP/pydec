import pydec
from pydec import Composition
import torch
import logging

torch.manual_seed(114514)

# t = torch.randn((3,))
# t2 = torch.zeros((2,))
# t3 = t2
# print(t2._version)
# print(torch.cat([t, t], dim=0, out=t2))
# print(t2._version)
# print(t2)
# print(t3)
# # # t = t + 1
# # print(t)
# # torch.tensor(t).unsqueeze_(1)
# # t.clone().detach().unsqueeze_(1)
# # # t3.squeeze_(1)
# pydec.set_bias_decomposition_func("abs_decomposition")
# c = Composition((2, 3), 3, dtype=torch.float)
# # c = Composition(torch.empty((3, 2)), torch.empty((4,)))
# # c = Composition(c)
# # print(c)
# t0 = torch.randn((2, 3), requires_grad=True)
# t1 = torch.randn((2, 3), requires_grad=True)
# t2 = torch.randn((2, 3), requires_grad=True)
# index = torch.LongTensor([[2, 1], [2, 0]])

# c[0] = t0
# c[1] = t1
# c[2] = t2
# print(c.size(1))
# # print(c.all(1, 2, keepdim=True))
# # torch.nn.Conv2d
# # print(t0)
# # print(t1)
# # print(t0.scatter(dim=1, index=index, src=t1, reduce="add")+t1+t2)
# # print(c.scatter(dim=1, index=index, src=t1, reduce="add").c_sum())
# # exit()
# # c._composition_tensor.requires_grad_(True)
# # c._composition_tensor += 3
# c = 2 * c
# c = c + 3
# # print(c[0])
# # c += 3
# # print(c)
# # c += 3
# c = c.permute(dims=(-1, 0))
# t0.permute(dims=(-1, 0))
# exit()
# # c = c.contiguous().view((6,)).to(c)

# # c = c.view_as(t0)
# # print(c.c_sum().sum())
# # print(c)
# # c = c.sum((-1,0), keepdim=True)
# # print(c)
# print(c.size())
# print(c)

# loss = c.sum().c_sum()
# loss.backward()
# print(t0.grad)
# # print(c.size())
# # c.unsqueeze_(1)
# # print(c.size())


def init_composition(size, c_num=3, requires_grad=False):
    c = Composition(size, c_num, dtype=torch.float)
    for i in range(c_num):
        c[i] = torch.randn(size, requires_grad=requires_grad)
    return c


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
        from pydec import set_bias_decomposition_func

        self.c[:] = 1.5
        set_bias_decomposition_func("abs_decomposition")
        c = self.c + 3
        assert (self.c._composition_tensor + 1 == c._composition_tensor).all()

        set_bias_decomposition_func("norm_decomposition")
        c = self.c + 3
        assert (self.c._composition_tensor + 1 == c._composition_tensor).all()

        set_bias_decomposition_func("hybrid_decomposition")
        c = self.c + 3
        assert (self.c._composition_tensor + 1 == c._composition_tensor).all()

        set_bias_decomposition_func("sign_decomposition")
        c = self.c + 3
        assert (self.c._composition_tensor + 1 == c._composition_tensor).all()

        set_bias_decomposition_func("hybrid_decomposition_threshold")
        c = self.c + 3
        assert (self.c._composition_tensor + 1 == c._composition_tensor).all()

        with pydec.no_bias_decomposition():
            c = self.c + 3
            assert (self.c._composition_tensor == c._composition_tensor).all()
            assert (self.c._residual_tensor + 3 == c._residual_tensor).all()

        with pydec.using_bias_decomposition_func("abs_decomposition"):
            self.c._composition_tensor[1] = -self.c._composition_tensor[1]
            c = self.c + 3
            assert (self.c._composition_tensor + 1 == c._composition_tensor).all()

    def test2(self):
        c = init_composition((2, 3))
        out = c + self.c
        assert torch.all(
            self.c._composition_tensor + c._composition_tensor
            == out._composition_tensor
        )


# input = torch.randn((16, 20, 512))

# c = pydec.Composition((16, 20, 512), component_num=20*512)
# c = c.view(16, 20*512)
# c = pydec.diagonal_init(c, src=input.view(16,20*512), dim=1)
# c = c.view_as(input)
# torch.zeros(requires_grad=)
# import torch.nn as nn
# from torch import Tensor


# class NN(nn.Module):
#     def __init__(self) -> None:
#         ...

#     def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
#         x1 = self.linear1(x1)
#         x1 = self.relu(x1)

#         x2 = self.linear2(x2)
#         x2 = self.relu(x2)

#         out = self.linear3(x1 + x2)
#         return out


# class NN(nn.Module):
#     def __init__(self) -> None:
#         self.linear1: nn.Linear = None
#         self.linear2: nn.Linear = None
#         self.linear3: nn.Linear = None

#     def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
#         x1 = self.linear1(x1)
#         x1 = self.relu(x1)

#         x2 = self.linear2(x2)
#         x2 = self.relu(x2)

#         out = self.linear3(x1 + x2)


#         # Initialize composition
#         import pydec
#         from pydec import Composition
#         c1 = Composition(x1.size(), component_num=2).to(x1)
#         c1[0] = x1

#         c2 = Composition(x2.size(), component_num=2).to(x2)
#         c2[1] = x2

#         # Apply the same operation for composition
#         c1 = pydec.nn.functional.linear(
#             c1, weight=self.linear1.weight, bias=self.linear1.bias
#         )
#         c1 = pydec.nn.functional.relu(c1)

#         c2 = pydec.nn.functional.linear(
#             c2, weight=self.linear2.weight, bias=self.linear2.bias
#         )
#         c2 = pydec.nn.functional.relu(c2)

#         c_out = pydec.nn.functional.linear3(
#             c1 + c2, weight=self.linear3.weight, bias=self.linear3.bias
#         )
#         return out, c_out

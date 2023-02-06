import pydec
from pydec import Composition
import torch


def init_composition(size, c_num=3, requires_grad=False):
    c = pydec.zeros(size, c_num, dtype=torch.float)
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


class TestLegacyRelu:
    def test_none(self):
        input = init_composition((3, 4))
        input._residual_tensor = torch.randn((3, 4))
        legacy_relu = pydec.nn.functional.legacy_relu
        with pydec.using_decomposition_func("none"):
            out = legacy_relu(input)

        ref = input.c_sum()
        zero_mask = ref < 0
        ref_out = input.masked_fill(zero_mask, 0.0)
        assert (out._composition_tensor == ref_out._composition_tensor).all()
        assert (out._residual_tensor == ref_out._residual_tensor).all()

    def test_hybrid(self):
        def hybrid_decomposition_(
            sum_value,
            context: Composition,
            *,
            threshold=0.15,
            eps=1e-6,
        ) -> Composition:

            composition = context._composition_tensor
            sum_composition = composition.sum(dim=0)
            abs_composition = composition.abs()
            abs_sum_composition = abs_composition.sum(dim=0, keepdim=True)
            instability_ratio = sum_composition.abs() / abs_sum_composition
            mask = (instability_ratio < threshold).expand_as(composition)

            composition[mask] = composition[mask].abs()

            multiplier = sum_value / composition.sum(dim=0)
            context._composition_tensor *= multiplier
            context._residual_tensor = 0.0
            return context

        input = init_composition((3, 4))
        input._residual_tensor = torch.randn((3, 4))
        legacy_relu = pydec.nn.functional.legacy_relu
        with pydec.using_decomposition_func("hybrid_decomposition"):
            with pydec.using_decomposition_args(threshold=0.15):
                out = legacy_relu(input)

        ref = input.c_sum()
        zero_mask = ref < 0
        ref_out = input.masked_fill(zero_mask, 0.0)
        residual_out = torch.nn.functional.relu(input._residual_tensor)
        masked_residual_out = ref_out._residual_tensor
        delta_out = hybrid_decomposition_(
            masked_residual_out - residual_out, input, threshold=0.15
        )
        ref_out += delta_out
        ref_out._residual_tensor = residual_out

        assert (out._composition_tensor == ref_out._composition_tensor).all()
        assert (out._residual_tensor == ref_out._residual_tensor).all()

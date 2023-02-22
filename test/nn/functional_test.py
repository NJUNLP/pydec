import pydec
from pydec import Composition
import torch
from test._composition_test import init_composition


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
        with pydec.core.decOVF.using_decomposition_func("none"):
            out = legacy_relu(input)

        ref = input.c_sum()
        zero_mask = ref < 0
        ref_out = input.masked_fill(zero_mask, 0.0)
        assert (out._component_tensor == ref_out._component_tensor).all()
        assert (out._residual_tensor == ref_out._residual_tensor).all()

    def test_hybrid(self):
        def hybrid_decomposition_(
            sum_value,
            context: Composition,
            *,
            threshold=0.15,
            eps=1e-6,
        ) -> Composition:

            composition = context._component_tensor
            sum_composition = composition.sum(dim=0)
            abs_composition = composition.abs()
            abs_sum_composition = abs_composition.sum(dim=0, keepdim=True)
            instability_ratio = sum_composition.abs() / abs_sum_composition
            mask = (instability_ratio < threshold).expand_as(composition)

            composition[mask] = composition[mask].abs()

            multiplier = sum_value / composition.sum(dim=0)
            context._component_tensor *= multiplier
            context._residual_tensor = 0.0
            return context

        input = init_composition((3, 4))
        input._residual_tensor = torch.randn((3, 4))
        legacy_relu = pydec.nn.functional.legacy_relu
        with pydec.core.decOVF.using_decomposition_func("hybrid_decomposition"):
            with pydec.core.decOVF.using_decomposition_args(threshold=0.15):
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

        assert (out._component_tensor == ref_out._component_tensor).all()
        assert (out._residual_tensor == ref_out._residual_tensor).all()


class TestRNN:
    single_input = torch.rand((3, 3))
    batch_input = torch.rand((3, 2, 3))
    batch_first_input = torch.rand((2, 3, 3))

    def test_1layer(self):
        rnn = torch.nn.RNN(
            input_size=3,
            hidden_size=4,
            num_layers=1,
            bias=True,
            dropout=0,
            bidirectional=False,
            nonlinearity="relu",
        )
        input = TestRNN.single_input
        c_input = pydec.zeros(input.size(), c_num=3)
        c_input = pydec.diagonal_init(c_input, input, 0)
        ref_out, ref_h = rnn(input)
        out, h = rnn(c_input)
        assert (out.c_sum() - ref_out).abs().sum() < 1e-2
        assert (h.c_sum() - ref_h).abs().sum() < 1e-2

        input = TestRNN.batch_input
        c_input = pydec.zeros(input.size(), c_num=3)
        c_input = pydec.diagonal_init(c_input, input, 0)
        ref_out, ref_h = rnn(input)
        out, h = rnn(c_input)
        assert (out.c_sum() - ref_out).abs().sum() < 1e-2
        assert (h.c_sum() - ref_h).abs().sum() < 1e-2

    def test_1layer_bi(self):
        rnn = torch.nn.RNN(
            input_size=3,
            hidden_size=4,
            num_layers=1,
            bias=True,
            dropout=0,
            bidirectional=True,
            nonlinearity="relu",
        )
        input = TestRNN.single_input
        c_input = pydec.zeros(input.size(), c_num=3)
        c_input = pydec.diagonal_init(c_input, input, 0)
        ref_out, ref_h = rnn(input)
        out, h = rnn(c_input)
        assert (out.c_sum() - ref_out).abs().sum() < 1e-2
        assert (h.c_sum() - ref_h).abs().sum() < 1e-2

        input = TestRNN.batch_input
        c_input = pydec.zeros(input.size(), c_num=3)
        c_input = pydec.diagonal_init(c_input, input, 0)
        ref_out, ref_h = rnn(input)
        out, h = rnn(c_input)
        assert (out.c_sum() - ref_out).abs().sum() < 1e-2
        assert (h.c_sum() - ref_h).abs().sum() < 1e-2

    def test_2layer(self):
        rnn = torch.nn.RNN(
            input_size=3,
            hidden_size=4,
            num_layers=2,
            bias=True,
            dropout=0,
            bidirectional=False,
            nonlinearity="relu",
        )
        input = TestRNN.single_input
        c_input = pydec.zeros(input.size(), c_num=3)
        c_input = pydec.diagonal_init(c_input, input, 0)
        ref_out, ref_h = rnn(input)
        out, h = rnn(c_input)
        assert (out.c_sum() - ref_out).abs().sum() < 1e-2
        assert (h.c_sum() - ref_h).abs().sum() < 1e-2

        input = TestRNN.batch_input
        c_input = pydec.zeros(input.size(), c_num=3)
        c_input = pydec.diagonal_init(c_input, input, 0)
        ref_out, ref_h = rnn(input)
        out, h = rnn(c_input)
        assert (out.c_sum() - ref_out).abs().sum() < 1e-2
        assert (h.c_sum() - ref_h).abs().sum() < 1e-2

    def test_2layer_bi(self):
        rnn = torch.nn.RNN(
            input_size=3,
            hidden_size=4,
            num_layers=2,
            bias=True,
            dropout=0,
            bidirectional=True,
            nonlinearity="relu",
        )
        input = TestRNN.single_input
        c_input = pydec.zeros(input.size(), c_num=3)
        c_input = pydec.diagonal_init(c_input, input, 0)
        ref_out, ref_h = rnn(input)
        out, h = rnn(c_input)
        assert (out.c_sum() - ref_out).abs().sum() < 1e-2
        assert (h.c_sum() - ref_h).abs().sum() < 1e-2

        input = TestRNN.batch_input
        c_input = pydec.zeros(input.size(), c_num=3)
        c_input = pydec.diagonal_init(c_input, input, 0)
        ref_out, ref_h = rnn(input)
        out, h = rnn(c_input)
        assert (out.c_sum() - ref_out).abs().sum() < 1e-2
        assert (h.c_sum() - ref_h).abs().sum() < 1e-2

    def test_1layer_packed(self):
        rnn = torch.nn.RNN(
            input_size=3,
            hidden_size=4,
            num_layers=1,
            bias=True,
            dropout=0,
            bidirectional=False,
            nonlinearity="relu",
        )

        input = TestRNN.batch_input
        c_input = pydec.zeros(input.size(), c_num=3)
        c_input = pydec.diagonal_init(c_input, input, 0)
        lengths = torch.tensor([2, 3], dtype=torch.long)
        input = torch.nn.utils.rnn.pack_padded_sequence(
            input, lengths, enforce_sorted=False
        )
        c_input = torch.nn.utils.rnn.pack_padded_sequence(
            c_input, lengths, enforce_sorted=False
        )
        ref_out, ref_h = rnn(input)
        out, h = rnn(c_input)

        assert (h.c_sum() - ref_h).abs().sum() < 1e-2
        assert (out.data.c_sum() - ref_out.data).abs().sum() < 1e-2

        ref_pad_out, ref_pad_lengths = torch.nn.utils.rnn.pad_packed_sequence(ref_out)
        pad_out, pad_lengths = torch.nn.utils.rnn.pad_packed_sequence(out)

        assert (pad_out.c_sum() - ref_pad_out).abs().sum() < 1e-2
        assert torch.all(ref_pad_lengths == pad_lengths)

    def test_2layer_packed(self):
        rnn = torch.nn.RNN(
            input_size=3,
            hidden_size=4,
            num_layers=2,
            bias=True,
            dropout=0,
            bidirectional=False,
            nonlinearity="relu",
        )

        input = TestRNN.batch_input
        c_input = pydec.zeros(input.size(), c_num=3)
        c_input = pydec.diagonal_init(c_input, input, 0)
        lengths = torch.tensor([2, 3], dtype=torch.long)
        input = torch.nn.utils.rnn.pack_padded_sequence(
            input, lengths, enforce_sorted=False
        )
        c_input = torch.nn.utils.rnn.pack_padded_sequence(
            c_input, lengths, enforce_sorted=False
        )
        ref_out, ref_h = rnn(input)
        out, h = rnn(c_input)

        assert (h.c_sum() - ref_h).abs().sum() < 1e-2
        assert (out.data.c_sum() - ref_out.data).abs().sum() < 1e-2

        ref_pad_out, ref_pad_lengths = torch.nn.utils.rnn.pad_packed_sequence(ref_out)
        pad_out, pad_lengths = torch.nn.utils.rnn.pad_packed_sequence(out)

        assert (pad_out.c_sum() - ref_pad_out).abs().sum() < 1e-2
        assert torch.all(ref_pad_lengths == pad_lengths)

    def test_2layer_bi_packed(self):
        rnn = torch.nn.RNN(
            input_size=3,
            hidden_size=4,
            num_layers=2,
            bias=True,
            dropout=0,
            bidirectional=True,
            nonlinearity="relu",
        )

        input = TestRNN.batch_input
        c_input = pydec.zeros(input.size(), c_num=3)
        c_input = pydec.diagonal_init(c_input, input, 0)
        lengths = torch.tensor([2, 3], dtype=torch.long)
        input = torch.nn.utils.rnn.pack_padded_sequence(
            input, lengths, enforce_sorted=False
        )
        c_input = torch.nn.utils.rnn.pack_padded_sequence(
            c_input, lengths, enforce_sorted=False
        )
        ref_out, ref_h = rnn(input)
        out, h = rnn(c_input)

        assert (h.c_sum() - ref_h).abs().sum() < 1e-2
        assert (out.data.c_sum() - ref_out.data).abs().sum() < 1e-2

        ref_pad_out, ref_pad_lengths = torch.nn.utils.rnn.pad_packed_sequence(ref_out)
        pad_out, pad_lengths = torch.nn.utils.rnn.pad_packed_sequence(out)

        assert (pad_out.c_sum() - ref_pad_out).abs().sum() < 1e-2
        assert torch.all(ref_pad_lengths == pad_lengths)


class TestLayerNorm:
    def test1(self):
        c_input = init_composition((2, 3, 4))
        input = c_input.c_sum()

        ref = torch.nn.functional.layer_norm(input, normalized_shape=[4])
        c_out = torch.nn.functional.layer_norm(c_input, normalized_shape=[4])
        assert (ref - c_out.c_sum()).abs().sum() < 1e-3

        ref = torch.nn.functional.layer_norm(input, normalized_shape=[3, 4])
        c_out = torch.nn.functional.layer_norm(c_input, normalized_shape=[3, 4])
        assert (ref - c_out.c_sum()).abs().sum() < 1e-3

    def test2(self):
        c_input = init_composition((2, 3, 4))
        input = c_input.c_sum()

        layernorm = torch.nn.LayerNorm((4,))
        ref = torch.nn.functional.layer_norm(
            input, normalized_shape=[4], weight=layernorm.weight, bias=layernorm.bias
        )
        c_out = torch.nn.functional.layer_norm(
            c_input, normalized_shape=[4], weight=layernorm.weight, bias=layernorm.bias
        )
        assert (ref - c_out.c_sum()).abs().sum() < 1e-3

        layernorm = torch.nn.LayerNorm(
            (
                3,
                4,
            )
        )
        ref = torch.nn.functional.layer_norm(
            input, normalized_shape=[3, 4], weight=layernorm.weight, bias=layernorm.bias
        )
        c_out = torch.nn.functional.layer_norm(
            c_input,
            normalized_shape=[3, 4],
            weight=layernorm.weight,
            bias=layernorm.bias,
        )
        assert (ref - c_out.c_sum()).abs().sum() < 1e-3

import pydec
from ._composition_test import init_composition
import pydec.core.decOVF as devOVF


class TestBiasArgs:
    c = init_composition((2, 3))

    def test_set_args1(self):
        args = {"eps": 1e-2, "arg1": "foo"}
        devOVF.set_decomposition_args(**args)
        bias_args = devOVF.get_decomposition_args()
        for k, v in args.items():
            assert bias_args[k] == v

        args = {"eps": 1e-3, "arg2": None}
        devOVF.set_decomposition_args(**args)
        bias_args = devOVF.get_decomposition_args()
        assert bias_args["arg1"] == "foo"
        for k, v in args.items():
            assert bias_args[k] == v

    def test_set_args2(self):
        args = {"eps": 1e-2, "arg1": "foo"}
        devOVF.set_decomposition_args(update=False, **args)
        bias_args = devOVF.get_decomposition_args()
        assert len(bias_args) == len(args)
        for k, v in args.items():
            assert bias_args[k] == v

        args = {"eps": 1e-3, "arg2": None}
        devOVF.set_decomposition_args(update=False, **args)
        bias_args = devOVF.get_decomposition_args()
        assert len(bias_args) == len(args)
        for k, v in args.items():
            assert bias_args[k] == v

    def test_using_args1(self):
        args = {"threshold": 0.8}
        devOVF.set_decomposition_func("hybrid_decomposition")
        assert "threshold" not in devOVF.get_decomposition_args()
        with devOVF.using_decomposition_args(**args):
            assert devOVF.get_decomposition_args()["threshold"] == args["threshold"]
            c = self.c + 3

        assert "threshold" not in devOVF.get_decomposition_args()

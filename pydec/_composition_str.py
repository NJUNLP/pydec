import torch
from torch._tensor_str import _tensor_str, _add_suffixes, PRINT_OPTS
from typing import List, Tuple


def _c_add_suffixes(composition_str, suffixes, indent, force_newline):
    tensor_strs = [composition_str]
    last_line_len = len(composition_str) - composition_str.rfind("\n") + 1
    for suffix in suffixes:
        suffix_len = len(suffix)
        if force_newline or last_line_len + suffix_len + 2 > PRINT_OPTS.linewidth:
            tensor_strs.append(",\n" + " " * indent + suffix)
            last_line_len = indent + suffix_len
            force_newline = False
        else:
            tensor_strs.append(", " + suffix)
            last_line_len += suffix_len + 2
    tensor_strs.append("}")
    return "".join(tensor_strs)


def _tensor_str_intern(inp, *, tensor_contents=None, prefix_indent=0):
    is_plain_tensor = type(inp) is torch.Tensor or type(inp) is torch.nn.Parameter
    if inp.is_nested:
        prefix = "nested_tensor("
    elif is_plain_tensor:
        prefix = "tensor("
    else:
        prefix = f"{type(inp).__name__}("

    prefix = " " * prefix_indent + prefix  # add by pydec

    indent = len(prefix)
    suffixes = []
    custom_contents_provided = tensor_contents is not None
    if custom_contents_provided:
        tensor_str = tensor_contents

    # This is used to extract the primal value and thus disable the forward AD
    # within this function.
    # TODO(albanD) This needs to be updated when more than one level is supported
    self, tangent = torch.autograd.forward_ad.unpack_dual(inp)

    # Note [Print tensor device]:
    # A general logic here is we only print device when it doesn't match
    # the device specified in default tensor type.
    # Currently torch.set_default_tensor_type() only supports CPU/CUDA, thus
    # torch._C._get_default_device() only returns either cpu or cuda.
    # In other cases, we don't have a way to set them as default yet,
    # and we should always print out device for them.
    if (
        self.device.type != torch._C._get_default_device()
        or (
            self.device.type == "cuda"
            and torch.cuda.current_device() != self.device.index
        )
        or (self.device.type == "mps")
    ):
        suffixes.append("device='" + str(self.device) + "'")

    # Tensor printing performs tensor operations like slice, indexing, etc to make it in a
    # representable format. These operations on xla/lazy tensor results in compilations. Hence,
    # to avoid compilations, copying the tensor to cpu before printing.
    if self.device.type == "xla" or self.device.type == "lazy":
        self = self.to("cpu")

    # TODO: add an API to map real -> complex dtypes
    _default_complex_dtype = (
        torch.cdouble if torch.get_default_dtype() == torch.double else torch.cfloat
    )
    has_default_dtype = self.dtype in (
        torch.get_default_dtype(),
        _default_complex_dtype,
        torch.int64,
        torch.bool,
    )
    if self.is_sparse:
        suffixes.append("size=" + str(tuple(self.shape)))
        suffixes.append("nnz=" + str(self._nnz()))
        if not has_default_dtype:
            suffixes.append("dtype=" + str(self.dtype))
        if not custom_contents_provided:
            indices_prefix = "indices=tensor("
            indices = self._indices().detach()
            indices_str = _tensor_str(indices, indent + len(indices_prefix))
            if indices.numel() == 0:
                indices_str += ", size=" + str(tuple(indices.shape))
            values_prefix = "values=tensor("
            values = self._values().detach()
            values_str = _tensor_str(values, indent + len(values_prefix))
            if values.numel() == 0:
                values_str += ", size=" + str(tuple(values.shape))
            tensor_str = (
                indices_prefix
                + indices_str
                + "),\n"
                + " " * indent
                + values_prefix
                + values_str
                + ")"
            )
    elif self.layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }:
        suffixes.append("size=" + str(tuple(self.shape)))
        suffixes.append("nnz=" + str(self._nnz()))
        if not has_default_dtype:
            suffixes.append("dtype=" + str(self.dtype))
        if not custom_contents_provided:
            compressed_indices_method, plain_indices_method = {
                torch.sparse_csr: (torch.Tensor.crow_indices, torch.Tensor.col_indices),
                torch.sparse_csc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices),
                torch.sparse_bsr: (torch.Tensor.crow_indices, torch.Tensor.col_indices),
                torch.sparse_bsc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices),
            }[self.layout]
            if self.layout in {torch.sparse_csr, torch.sparse_bsr}:
                cdimname, pdimname = "row", "column"
            else:
                cdimname, pdimname = "column", "row"
            compressed_indices_prefix = f"c{cdimname[:3]}_indices=tensor("
            compressed_indices = compressed_indices_method(self).detach()
            compressed_indices_str = _tensor_str(
                compressed_indices, indent + len(compressed_indices_prefix)
            )
            if compressed_indices.numel() == 0:
                compressed_indices_str += ", size=" + str(
                    tuple(compressed_indices.shape)
                )
            plain_indices_prefix = f"{pdimname[:3]}_indices=tensor("
            plain_indices = plain_indices_method(self).detach()
            plain_indices_str = _tensor_str(
                plain_indices, indent + len(plain_indices_prefix)
            )
            if plain_indices.numel() == 0:
                plain_indices_str += ", size=" + str(tuple(plain_indices.shape))
            values_prefix = "values=tensor("
            values = self.values().detach()
            values_str = _tensor_str(values, indent + len(values_prefix))
            if values.numel() == 0:
                values_str += ", size=" + str(tuple(values.shape))
            tensor_str = (
                compressed_indices_prefix
                + compressed_indices_str
                + "),\n"
                + " " * indent
                + plain_indices_prefix
                + plain_indices_str
                + "),\n"
                + " " * indent
                + values_prefix
                + values_str
                + ")"
            )
    elif self.is_quantized:
        suffixes.append("size=" + str(tuple(self.shape)))
        if not has_default_dtype:
            suffixes.append("dtype=" + str(self.dtype))
        suffixes.append("quantization_scheme=" + str(self.qscheme()))
        if (
            self.qscheme() == torch.per_tensor_affine
            or self.qscheme() == torch.per_tensor_symmetric
        ):
            suffixes.append("scale=" + str(self.q_scale()))
            suffixes.append("zero_point=" + str(self.q_zero_point()))
        elif (
            self.qscheme() == torch.per_channel_affine
            or self.qscheme() == torch.per_channel_symmetric
            or self.qscheme() == torch.per_channel_affine_float_qparams
        ):
            suffixes.append("scale=" + str(self.q_per_channel_scales()))
            suffixes.append("zero_point=" + str(self.q_per_channel_zero_points()))
            suffixes.append("axis=" + str(self.q_per_channel_axis()))
        if not custom_contents_provided:
            tensor_str = _tensor_str(self.dequantize(), indent)
    elif self.is_nested:
        if not custom_contents_provided:

            def indented_str(s, indent):
                return "\n".join(f"  {line}" for line in s.split("\n"))

            strs = ",\n".join(
                indented_str(str(t), indent + 1)
                for t in torch.ops.aten.unbind.int(self, 0)
            )
            tensor_str = f"[\n{strs}\n]"
    else:
        if self.is_meta:
            suffixes.append("size=" + str(tuple(self.shape)))
            if self.dtype != torch.get_default_dtype():
                suffixes.append("dtype=" + str(self.dtype))
            # TODO: This implies that ellipses is valid syntax for allocating
            # a meta tensor, which it could be, but it isn't right now
            if not custom_contents_provided:
                tensor_str = "..."
        else:
            if self.numel() == 0 and not self.is_sparse:
                # Explicitly print the shape if it is not (0,), to match NumPy behavior
                if self.dim() != 1:
                    suffixes.append("size=" + str(tuple(self.shape)))

                # In an empty tensor, there are no elements to infer if the dtype
                # should be int64, so it must be shown explicitly.
                if self.dtype != torch.get_default_dtype():
                    suffixes.append("dtype=" + str(self.dtype))
                if not custom_contents_provided:
                    tensor_str = "[]"
            else:
                if not has_default_dtype:
                    suffixes.append("dtype=" + str(self.dtype))

                if not custom_contents_provided:
                    if self.layout != torch.strided:
                        tensor_str = _tensor_str(self.to_dense(), indent)
                    else:
                        tensor_str = _tensor_str(self, indent)

    if self.layout != torch.strided:
        suffixes.append("layout=" + str(self.layout))

    # Use inp here to get the original grad_fn and not the one generated by the forward grad
    # unpacking.
    if inp.grad_fn is not None:
        name = type(inp.grad_fn).__name__
        if name == "CppFunction":
            name = inp.grad_fn.name().rsplit("::", 1)[-1]
        suffixes.append("grad_fn=<{}>".format(name))
    elif inp.requires_grad:
        suffixes.append("requires_grad=True")

    if self.has_names():
        suffixes.append("names={}".format(self.names))

    if tangent is not None:
        suffixes.append("tangent={}".format(tangent))

    # pydec: ignore the suffixes and return it
    string_repr = _add_suffixes(
        prefix + tensor_str, [], indent, force_newline=self.is_sparse
    )

    # Check if this instance is flagged as a parameter and change the repr accordingly.
    # Unfortunately, this function has to be aware of this detail.
    # NB: This is currently skipped for plain tensor parameters to maintain BC. In the future,
    # this should be done for those as well to produce a valid repr.
    if isinstance(self, torch.nn.Parameter) and not is_plain_tensor:
        # pydec: move the prefix indent to the beginning
        string_repr = " " * prefix_indent + f"Parameter({string_repr[prefix_indent:]})"

    return string_repr, suffixes


def _t_str(
    self: torch.Tensor, *, tensor_contents: str = None, prefix_indent=0
) -> Tuple:
    with torch.no_grad():
        return _tensor_str_intern(
            self, tensor_contents=tensor_contents, prefix_indent=prefix_indent
        )


def _c_str(self, *, composition_contents: List[str] = None):
    """
    Args:
        self: the composition to be represented
        composition_contents: should be a list of string and the length of the list should be
        equal to the number of components (include the residual). For components that do not
        use customized content, use `None` placeholder in the list.
    """
    if composition_contents is not None:
        contents_num = len(composition_contents)
        components_num = self.numc() + 1
        if contents_num != components_num:
            if contents_num < components_num:
                composition_contents.extend([None] * (self.numc() + 1 - contents_num))
            else:
                # ignore the redundant
                composition_contents = composition_contents[:components_num]
    prefix = "composition{"
    indent = 2  # fix to 2 looks nice

    components_reprs = [
        _t_str(
            self[i],
            tensor_contents=composition_contents[i]
            if composition_contents is not None
            else None,
            prefix_indent=indent + 2,  # second indent
        )[0]
        for i in range(len(self))
    ]
    residual_repr, suffixes = _t_str(
        self._residual_tensor,
        tensor_contents=composition_contents[-1]
        if composition_contents is not None
        else None,
        prefix_indent=indent + 2,  # second indent
    )

    components_str = " " * indent + "components:\n"
    components_str += ",\n".join(components_reprs)

    residual_str = " " * indent + "residual:\n"
    residual_str += residual_repr

    string_repr = _c_add_suffixes(
        prefix + "\n" + components_str + ",\n" + residual_str,
        suffixes,
        indent,
        force_newline=True,
    )

    return string_repr

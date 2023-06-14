# PYDEC.BMM
> pydec.bmm(input, mat2, *, out=None) →  {{{pydec_Composition}}}

Performs a batch matrix-matrix product of matrices stored in `input` and `mat2`.

`input` and `mat2` must be 3-D compositions/tensors each containing the same number of matrices. `input` and `mat2` can't both be compositions. If `input` is a composition, then `mat2` can only be a tensor, and vice versa.

If `input` is a $(b\times n\times m)$ composition/tensor, `mat2` is a $(b\times m\times p)$ composition/tensor, `out` will be a $(b\times n\times p)$ composition.

?> This function does not [broadcast](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics). For broadcasting matrix products, see {{#auto_link}}pydec.matmul{{/auto_link}}.


**Parameters:**

* **input** (*{{{pydec_Composition}}} or {{{torch_Tensor}}}*) – the first batch of matrices to be multiplied.
* **mat2** (*{{{pydec_Composition}}} or {{{torch_Tensor}}}*) - the second batch of matrices to be multiplied.

**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> mat1 = pydec.Composition(torch.randn(2, 10, 2, 3))
>>> mat2 = torch.randn(10, 3, 3)
>>> out = pydec.bmm(mat1, mat2)
>>> out.size()
"""
torch.Size([10, 2, 3])
"""
>>> out.c_size()
"""
torch.Size([2, 10, 2, 3])
"""
```

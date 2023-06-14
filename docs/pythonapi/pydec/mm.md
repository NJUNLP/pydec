# PYDEC.MM
> pydec.mm(input, mat2, *, out=None) →  {{{pydec_Composition}}}

Performs a matrix multiplication of the matrices `input` and `mat2`.

`input` and `mat2` can't both be compositions. If `input` is a composition, then `mat2` can only be a tensor, and vice versa.

If `input` is a $(n\times m)$ composition/tensor, `mat2` is a $(m\times p)$ composition/tensor, `out` will be a $(n\times p)$ composition.

?> This function does not [broadcast](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics). For broadcasting matrix products, see {{#auto_link}}pydec.matmul{{/auto_link}}.


**Parameters:**

* **input** (*{{{pydec_Composition}}} or {{{torch_Tensor}}}*) – the first matrix to be matrix multiplied.
* **mat2** (*{{{pydec_Composition}}} or {{{torch_Tensor}}}*) - the second matrix to be matrix multiplied.

**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> mat1 = pydec.Composition(torch.randn(2, 2, 3))
>>> mat2 = torch.randn(3, 3)
>>> pydec.mm(mat1, mat2)
"""
composition{
  components:
    tensor([[-0.5583,  0.5535, -0.5491],
            [-0.8766,  1.1141, -0.9149]]),
    tensor([[0.0630, 1.0846, 0.6208],
            [0.6416, 0.0794, 0.6329]]),
  residual:
    tensor([[0., 0., 0.],
            [0., 0., 0.]])}
"""
```

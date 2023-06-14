# PYDEC.MV
> pydec.mv(input, vec, *, out=None) →  {{{pydec_Composition}}}

Performs a matrix-vector product of the matrix `input` and the vector `vec`.

`input` and `vec` can't both be compositions. If `input` is a composition, then `vec` can only be a tensor, and vice versa.

If `input` is a $(n\times m)$ composition/tensor, `vec` is a 1-D composition/tensor of size $m$, `out` will be 1-D of size $n$.

?> This function does not [broadcast](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics).


**Parameters:**

* **input** (*{{{pydec_Composition}}} or {{{torch_Tensor}}}*) – matrix to be multiplied.
* **vec** (*{{{pydec_Composition}}} or {{{torch_Tensor}}}*) - vector to be multiplied.

**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> mat = pydec.Composition(torch.randn(2, 2, 3))
>>> vec = torch.randn(3)
>>> pydec.mv(mat, vec)
"""
composition{
  components:
    tensor([ 0.4769, -0.3806]),
    tensor([1.1718, 0.4191]),
  residual:
    tensor([0., 0.])}
"""
```

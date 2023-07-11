# PYDEC.COMPOSITION.SCATTER_
> Composition.scatter_(dim, index, src, reduce=None) →  {{{pydec_Composition}}}

Writes all values from the tensor `src` into each component (including residual) of `self` at the indices specified in the `index` tensor. For each value in `src`, its output index is specified by its index in `src` for `dimension != dim` and by the corresponding value in `index` for `dimension = dim`.

For a 3-D composition, `self` is updated as:
```python
self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
self[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1
self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2
```

See [torch.Tensor.scatter_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html#torch.Tensor.scatter_) for tmore information about this function.


**Parameters:**

* **dim** (*{{{python_int}}}*) – the axis along which to index.
* **index** (*LongTensor*) – the indices of elements to scatter, can be either empty or of the same dimensionality as `src`. When empty, the operation returns `self` unchanged.
* **src** (*{{{torch_Tensor}}} or {{{python_float}}}*) - the source element(s) to scatter.
* **reduce** ({{{python_str}}}, optional) – reduction operation to apply, can be either *'add'* or *'multiply'*.

<!-- TODO: add examples -->
# PYDEC.MASKED_SELECT
> pydec.masked_select(input, mask, *, out=None) →  {{{pydec_Composition}}}

Returns a new 1-D composition which indexes the `input` composition according to the boolean mask `mask` which is a *BoolTensor*.

The shapes of the `mask` tensor and the `input` composition don’t need to match, but they must be broadcastable.

?> The returned composition does **not** use the same storage as the original composition.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **mask** (*BoolTensor*) – the tensor containing the binary mask to index with.

**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> c = pydec.Composition(torch.rand((2, 3, 4)))
>>> c
"""
composition{
  components:
    tensor([[0.1570, 0.9172, 0.2099, 0.4501],
            [0.4613, 0.9657, 0.7494, 0.9688],
            [0.7802, 0.4493, 0.7036, 0.9793]]),
    tensor([[0.3224, 0.7608, 0.7118, 0.8885],
            [0.3152, 0.8221, 0.6120, 0.6981],
            [0.3319, 0.5036, 0.5729, 0.6784]]),
  residual:
    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]])}
"""
>>> mask = c.ge(1.0)
>>> mask
"""
tensor([[False,  True, False,  True],
        [False,  True,  True,  True],
        [ True, False,  True,  True]])
"""
>>> pydec.masked_select(c, mask)
"""
composition{
  components:
    tensor([0.9172, 0.4501, 0.9657, 0.7494, 0.9688, 0.7802, 0.7036, 0.9793]),
    tensor([0.7608, 0.8885, 0.8221, 0.6120, 0.6981, 0.3319, 0.5729, 0.6784]),
  residual:
    tensor([0., 0., 0., 0., 0., 0., 0., 0.])}
"""
```

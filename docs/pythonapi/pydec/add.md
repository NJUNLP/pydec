# PYDEC.ADD
> pydec.add(input, other, *, alpha=1, out=None) →  {{{pydec_Composition}}}

Adds `other`, scaled by `alpha`, to `input`.

If `other` is a tensor or number, then it is added to `input.residual`.

$$
\text{out}^R_j=\text{input}^R_j+\text{alpha}\times\text{other}_j
$$

If `other` is a composition, then it should have the same number of components as the `input`, and the addition is performed separately between the components.

$$
\text{out}_{ij}=\text{input}_{ij}+\text{alpha}\times\text{other}_{ij}
$$


Supports [broadcasting to a common shape](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics), [type promotion](https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc), and integer or float inputs.

<!-- Not tested on complex inputs-->

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **other** (*{{{pydec_Composition}}}, {{{torch_Tensor}}}, or Number*) - the composition, tensor, or number to add to `input`.

**Keyword Arguments:**
* **alpha** (*Number*) - the multiplier for `other`.
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> a = pydec.Composition(torch.randn(3, 2))
>>> a
"""
composition{
  components:
    tensor([2.3399, 0.4669]),  
    tensor([1.0977, 0.3482]),  
    tensor([ 0.5099, -0.4043]),
  residual:
    tensor([0., 0.])}
"""
>>> pydec.add(a, 20)
"""
composition{
  components:
    tensor([2.3399, 0.4669]),
    tensor([1.0977, 0.3482]),
    tensor([ 0.5099, -0.4043]),
  residual:
    tensor([20., 20.])}
"""
>>> b = pydec.Composition(torch.randn(3, 2))
>>> b
"""
composition{
  components:
    tensor([0.2565, 0.6118]),
    tensor([-0.1889,  1.8855]),
    tensor([-1.7531,  0.5653]),
  residual:
    tensor([0., 0.])}
"""
>>> pydec.add(a, b, alpha=10)
"""
composition{
  components:
    tensor([4.9053, 6.5845]),
    tensor([-0.7916, 19.2029]),
    tensor([-17.0215,   5.2486]),
  residual:
    tensor([0., 0.])}
"""
```

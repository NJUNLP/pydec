# PYDEC.SUB
> pydec.sub(input, other, *, alpha=1, out=None) →  {{{pydec_Composition}}}

Subtracts `other`, scaled by `alpha`, from `input`.

If `other` is a tensor or number, then it is subtracted from `input.residual`.

$$
\text{out}^R_j=\text{input}^R_j-\text{alpha}\times\text{other}_j
$$

If `other` is a composition, then it should have the same number of components as the `input`, and the subtraction is performed separately between the components.

$$
\text{out}_{ij}=\text{input}_{ij}-\text{alpha}\times\text{other}_{ij}
$$


Supports [broadcasting to a common shape](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics), [type promotion](https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc), and integer or float inputs.

<!-- Not tested on complex inputs-->

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **other** (*{{{pydec_Composition}}}, {{{torch_Tensor}}}, or Number*) - the composition, tensor, or number to subtract from `input`.

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
    tensor([ 0.4552, -1.6567]),
    tensor([-1.2678, -0.1593]),
    tensor([-0.6507,  0.4133]),
  residual:
    tensor([0., 0.])}
"""
>>> pydec.sub(a, 20)
"""
composition{
  components:
    tensor([ 0.4552, -1.6567]),
    tensor([-1.2678, -0.1593]),
    tensor([-0.6507,  0.4133]),
  residual:
    tensor([-20., -20.])}
"""
>>> b = pydec.Composition(torch.randn(3, 2))
>>> b
"""
composition{
  components:
    tensor([-0.1048, -0.3529]),
    tensor([-0.5995,  1.5224]),
    tensor([1.8653, 0.2317]),
  residual:
    tensor([0., 0.])}
"""
>>> pydec.sub(a, b, alpha=10)
"""
composition{
  components:
    tensor([1.5029, 1.8727]),
    tensor([  4.7275, -15.3832]),
    tensor([-19.3035,  -1.9040]),
  residual:
    tensor([0., 0.])}
"""
```

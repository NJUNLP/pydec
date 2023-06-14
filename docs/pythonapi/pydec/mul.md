# PYDEC.MUL
> pydec.mul(input, other, *, out=None) →  {{{pydec_Composition}}}

Multiplies each component in `input` by `other`.

$$
\text{out}_{ij}=\text{input}_{ij}\times\text{other}_j
$$

Supports [broadcasting to a common shape](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics), [type promotion](https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc), and integer or float inputs.

<!-- Not tested on complex inputs-->

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **other** (*{{{torch_Tensor}}} or Number*) - the tensor or number to multiply to `input`.

**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> a = pydec.Composition(torch.randn(3, 2), residual_tensor=torch.randn(2))
>>> a
"""
composition{
  components:
    tensor([-0.6434,  0.2326]),
    tensor([-1.0220,  0.4257]),
    tensor([-0.5581, -0.5223]),
  residual:
    tensor([ 1.4002, -0.7036])}
"""
>>> pydec.mul(a, 100)
"""
composition{
  components:
    tensor([-64.3426,  23.2617]),
    tensor([-102.2013,   42.5693]),
    tensor([-55.8086, -52.2251]),
  residual:
    tensor([140.0174, -70.3571])}
"""
>>> b = torch.randn(2, 1)
>>> b
"""
tensor([[-0.4458],
        [ 1.0107]])
"""
>>> pydec.mul(a, b)
"""
composition{
  components:
    tensor([[ 0.2869, -0.1037],
            [-0.6503,  0.2351]]),
    tensor([[ 0.4556, -0.1898],
            [-1.0330,  0.4303]]),
    tensor([[ 0.2488,  0.2328],
            [-0.5641, -0.5279]]),
  residual:
    tensor([[-0.6242,  0.3137],
            [ 1.4152, -0.7111]])}
"""
```

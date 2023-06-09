# PYDEC.DIV
> pydec.div(input, other, *, rounding_mode=None, out=None) →  {{{pydec_Composition}}}

Divides each element of every component in the `input` input by the corresponding element of `other`.

$$
\text{out}_{ij}=\frac{\text{input}_{ij}}{\text{other}_j}
$$

?> By default, this performs a "true" division like Python 3. See the `rounding_mode` argument for floor division.

Supports [broadcasting to a common shape](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics), [type promotion](https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc), and integer or float inputs. Always promotes integer types to the default scalar type.

<!-- Not tested on complex inputs-->

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the dividend.
* **other** (*{{{torch_Tensor}}} or Number*) - the divisor.

**Keyword Arguments:**
* **rounding_mode** (*{{{python_str}}}, optional*) - </br>Type of rounding applied to the result:
  * *None* - default behavior. Performs no rounding and, if both `input` and `other` are integer types, promotes the inputs to the default scalar type. Equivalent to true division in Python (the `/` operator) and NumPy's `np.true_divide`.
  * *"trunc"* - rounds the results of the division towards zero. Equivalent to C-style integer division.
  * *"floor"* - rounds the results of the division down. Equivalent to floor division in Python (the `//` operator) and NumPy's `np.floor_divide`.
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> a = pydec.Composition(torch.randn(3, 2), residual_tensor=torch.randn(2))
>>> a
"""
composition{
  components:
    tensor([ 0.3476, -1.6493]),
    tensor([-2.1204,  0.6018]),
    tensor([ 2.0818, -0.0750]),
  residual:
    tensor([-0.1288,  1.2147])}
"""
>>> pydec.div(a, 0.5)
"""
composition{
  components:
    tensor([ 0.6953, -3.2985]),
    tensor([-4.2408,  1.2036]),
    tensor([ 4.1635, -0.1500]),
  residual:
    tensor([-0.2576,  2.4294])}
"""
>>> b = torch.randn(2)
>>> b
"""
tensor([ 0.5074, -0.8636])
"""
>>> pydec.div(a, b, rounding_mode='trunc')
"""
composition{
  components:
    tensor([0., 1.]),
    tensor([-4., -0.]),
    tensor([4., 0.]),
  residual:
    tensor([-0., -1.])}
"""
>>> torch.div(a, b, rounding_mode='floor')
"""
composition{
  components:
    tensor([0., 1.]),
    tensor([-5., -1.]),
    tensor([4., 0.]),
  residual:
    tensor([-1., -2.])}
"""
```

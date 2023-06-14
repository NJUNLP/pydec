# PYDEC.C_CAT
> pydec.c_cat(compositions, dim=0, *, sum_residual=True, out=None) →  {{{pydec_Composition}}}

Concatenates the given sequence of `seq` compositions in the component dimension. All compositions must either have the same shape or be empty.

**Parameters:**

* **compositions** (*sequence of Compositions*) - any python sequence of compositions of the same type. Non-empty compositions provided must have the same shape.

**Keyword Arguments:**
* **sum_residual** (*{{{python_bool}}}, optional*) - If **True**, the residual of the returned composition is the sum of the residuals of all compositions, otherwise it is initialized to *0*. Default: **True**.
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> x = pydec.Composition(torch.randn(2,3))
>>> x += 1
>>> x
"""
composition{
  components:
    tensor([-1.8787, -0.2230, -0.3781]),
    tensor([ 0.5633, -0.3760,  1.4050]),
  residual:
    tensor([1., 1., 1.])}
"""
>>> pydec.c_cat((x, x, x))
"""
composition{
  components:
    tensor([-1.8787, -0.2230, -0.3781]),
    tensor([ 0.5633, -0.3760,  1.4050]),
    tensor([-1.8787, -0.2230, -0.3781]),
    tensor([ 0.5633, -0.3760,  1.4050]),
    tensor([-1.8787, -0.2230, -0.3781]),
    tensor([ 0.5633, -0.3760,  1.4050]),
  residual:
    tensor([3., 3., 3.])}
"""
>>> pydec.c_cat((x, x, x), sum_residual=False)
"""
composition{
  components:
    tensor([-1.8787, -0.2230, -0.3781]),
    tensor([ 0.5633, -0.3760,  1.4050]),
    tensor([-1.8787, -0.2230, -0.3781]),
    tensor([ 0.5633, -0.3760,  1.4050]),
    tensor([-1.8787, -0.2230, -0.3781]),
    tensor([ 0.5633, -0.3760,  1.4050]),
  residual:
    tensor([0., 0., 0.])}
"""
```

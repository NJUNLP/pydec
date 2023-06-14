# PYDEC.CAT
> pydec.cat(compositions, dim=0, *, out=None) →  {{{pydec_Composition}}}

Concatenates the given sequence of `seq` compositions in the given dimension. All compositions must either have the same shape and component number (except in the concatenating dimension) or be empty.

**Parameters:**

* **compositions** (*sequence of Compositions*) - any python sequence of compositions of the same type. Non-empty compositions provided must have the same shape and component number, except in the cat dimension.
* **dim** (*{{{python_int}}}, optional*) - the dimension over which the compositions are concatenated.

**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.


Example:
```python
>>> x = pydec.Composition(torch.randn(3,2))
>>> x
"""
composition{
  components:
    tensor([1.4950, 0.5403]),
    tensor([ 0.4686, -0.6612]),
    tensor([ 0.2949, -1.0854]),
  residual:
    tensor([0., 0.])}
"""
>>> pydec.cat((x, x, x), 0)
"""
composition{
  components:
    tensor([1.4950, 0.5403, 1.4950, 0.5403, 1.4950, 0.5403]),
    tensor([ 0.4686, -0.6612,  0.4686, -0.6612,  0.4686, -0.6612]),
    tensor([ 0.2949, -1.0854,  0.2949, -1.0854,  0.2949, -1.0854]),
  residual:
    tensor([0., 0., 0., 0., 0., 0.])}
"""
```

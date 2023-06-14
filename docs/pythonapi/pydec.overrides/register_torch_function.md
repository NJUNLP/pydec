# PYDEC.OVERRIDES.REGISTER_TORCH_FUNCTION
> pydec.overrides.register_torch_function(torch_function)

The decorator to register a customized function to handle invocations of `torch_function` on {{{pydec_Composition}}}.

?> For speed and flexibility the `__torch_function__` dispatch mechanism does not check that the signature of an override function matches the signature of the function being overrided in the torch API. For some applications ignoring optional arguments would be fine but to ensure full compatibility with Tensor, user implementations of torch API functions should take care to exactly emulate the API of the function that is being overrided. See [Extending torch](https://pytorch.org/docs/stable/notes/extending.html#extending-torch) for more details.

!> APIs under the `pydec` namespace will not be overridden.

**Parameters:**

* **torch_function** (*{{{python_callable}}}*) - the torch function to be overridden, must be overridable functions of torch.

Example:
```python
@pydec.overrides.register_torch_function(torch.sin)
def my_sin(input: Composition, *, out: Optional[Tensor] = None) -> Composition:
    return pydec.c_apply(input, torch.sin)

c = pydec.Composition(torch.randn(2, 3))

print(c,"\n\n")
print(torch.sin(c))
```

Out:
```python
"""
composition{
  components:
    tensor([-1.7797, -0.2707, -0.4457]),
    tensor([-1.3793, -0.3127, -0.2035]),
  residual:
    tensor([0., 0., 0.])} 


composition{
  components:
    tensor([-0.9783, -0.2674, -0.4311]),
    tensor([-0.9817, -0.3077, -0.2021]),
  residual:
    tensor([0., 0., 0.])}
"""
```
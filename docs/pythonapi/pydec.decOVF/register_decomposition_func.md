# PYDEC.DECOVF.REGISTER_DECOMPOSITION_FUNC
> pydec.decOVF.register_decomposition_func(name)

Decorator for registering a decomposition algorithm with `name`. Cannot be the same name of the already registered algorithms.

Your function signature should contain at least the positional arguments `input` and `func` and the keyword arguments `ref` and `inplace`:
```python
@pydec.decOVF.register_decomposition_func("your_algorithm_name")
def customized_decomposition(
    input: Composition,
    func: Callable[[Tensor], Tensor],
    *,
    ref: Optional[Tensor] = None,
    inplace: _bool = False,
) -> Composition:
    ...
```
where `input` is the input composition, `func` is an element-wise function whose input and output are both tensor, `ref` is the reference tensor of `input` (ground truth for `input.recovery`), and `inplace` indicates whether do the decomposition in-place.


You can add more keyword arguments to the function signature, which can be specified via {{#auto_link}}pydec.decOVF.set_decomposition_args{{/auto_link}} and {{#auto_link}}pydec.decOVF.using_decomposition_args{{/auto_link}}:
```python
@pydec.decOVF.register_decomposition_func("your_algorithm_name")
def customized_decomposition(input, func, *, ref=None, inplace=False, my_arg=None) -> Composition:
    ...
```

**Parameters:**

* **name** (*{{{python_str}}}*) – name of the algorithm to be registered.

**Raises:**
* {{{python_ValueError}}} - if the `name` is already registered.


Examples:
```python
@pydec.decOVF.register_decomposition_func("my_algorithm_name")
def customized_decomposition(
    input, func, *, ref=None, inplace=False, my_arg=None
) -> Composition:
    print("invoking customized decomposition!")
    print("my arg: {}".format(my_arg))
    return pydec.decOVF._none_decomposition(input, func, ref=ref, inplace=inplace)


pydec.decOVF.set_decomposition_func("my_algorithm_name")
pydec.decOVF.set_decomposition_args(my_arg=True, my_arg2="foo")
c = pydec.Composition(torch.randn(2, 3))
print(c, "\n\n")
print(torch.nn.functional.relu(c))
```

out:
```python
"""
composition{
  components:
    tensor([-0.7532, -0.1263,  0.9695]),
    tensor([ 0.2721,  0.8744, -0.4808]),
  residual:
    tensor([0., 0., 0.])} 


invoking customized decomposition!
my arg: True
composition{
  components:
    tensor([0., 0., 0.]),
    tensor([0., 0., 0.]),
  residual:
    tensor([0.0000, 0.7481, 0.4887])}
"""
```
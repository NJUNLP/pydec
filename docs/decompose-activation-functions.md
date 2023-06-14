# Decompose Activation Functions
As the activation functions are non-linear, linear substitutes need to be found to generate decomposition.


## Configuring the decomposition algorithm
PyDec has some built-in strategies to decompose activation functions. By default, PyDec uses the algorithm [*affine decomposition*]().

To use the specified algorithm, call {{#auto_link}}pydec.decOVF.set_decomposition_func{{/auto_link}} and pass in the name of the decomposition algorithm. This will set the global decomposition algorithm for PyDec. [Here](#decomposition-algorithms) is a list of all the built-in strategies and their names.

### Configuring by context

If you need to locally use the specified decomposition algorithm in a context, use {{#auto_link}}pydec.decOVF.using_decomposition_func{{/auto_link}} and pass in the name to create the context.

Example:
```python
pydec.decOVF.set_decomposition_func("affine")
# The code here uses the algorithm called 'affine'.
print(pydec.decOVF.get_decomposition_name()) # 'affine'
...
with pydec.decOVF.using_decomposition_func("scaling"):
    # The code here uses the algorithm called 'scaling'.
    print(pydec.decOVF.get_decomposition_name()) # 'scaling'
    ...
```

### Specify the arguments for decomposition algorithm

Some decomposition algorithm provide configurable hyperparameters, but the arguments cannot be passed explicitly when calling the torch operators.

Use {{#auto_link}}pydec.decOVF.set_decomposition_args{{/auto_link}} to set the arguments of the decomposition algorithm. We also provide the context manager to locally set arguments, by using {{#auto_link}}pydec.decOVF.using_decomposition_args{{/auto_link}}.

Example:
```python
pydec.decOVF.set_decomposition_args(threshold=0.1)
print(pydec.decOVF.get_decomposition_args()) # "{'threshold': 0.1}"
...
with pydec.decOVF.using_decomposition_args(threshold=0.2, foo="foo"):
    print(pydec.decOVF.get_decomposition_args()) # "{'threshold': 0.2, 'foo': 'foo'}"
    ...
```

## Decomposition algorithms
| Name          | API                                                                  | Comments                                                                                                 |
| ------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| affine        | {{#auto_link}}pydec.decOVF.affine_decomposition{{/auto_link}}        | The decomposition $\hat{\mathscr{D}}$(signed) in our paper.                                              |
| scaling       | {{#auto_link}}pydec.decOVF.scaling_decomposition{{/auto_link}}       | The decomposition $\bar{\mathscr{D}}$ in our paper.                                                      |
| abs_affine    | {{#auto_link}}pydec.decOVF.abs_affine_decomposition{{/auto_link}}    | The decomposition $\hat{\mathscr{D}}$(abs) in our paper.                                                 |
| hybrid_affine | {{#auto_link}}pydec.decOVF.hybrid_affine_decomposition{{/auto_link}} | $\hat{\mathscr{D}}$(signed) and $\hat{\mathscr{D}}$(abs) are hybridized by the hyperparameter $\lambda$. |
| none          | {{#auto_link}}pydec.decOVF._none_decomposition{{/auto_link}}         | No decomposition is performed.                                                                           |


## Customizing the decomposition algorithm

To use a customized decomposition algorithm, you need to register your function using {{#auto_link}}pydec.decOVF.register_decomposition_func{{/auto_link}} decorator:
```python
@pydec.decOVF.register_decomposition_func("your_algorithm_name")
def customized_decomposition(input, func, *, ref=None, inplace=False) -> Composition:
    ...
```
Your function signature should contain at least the positional arguments `input` and `func` and the keyword arguments `ref` and `inplace`. `input` is the input composition. `func` is an element-wise function whose input and output are both tensor, [`torch.nn.functional.relu()`](https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html#torch.nn.functional.relu) for example.

You can add more keyword arguments to the function signature, which can be specified via {{#auto_link}}pydec.decOVF.set_decomposition_args{{/auto_link}} and {{#auto_link}}pydec.decOVF.using_decomposition_args{{/auto_link}}.
```python
@pydec.decOVF.register_decomposition_func("your_algorithm_name")
def customized_decomposition(input, func, *, ref=None, inplace=False, my_arg=None) -> Composition:
    ...
```

Use {{#auto_link}}pydec.decOVF.set_decomposition_func{{/auto_link}} and {{#auto_link}}pydec.decOVF.using_decomposition_func{{/auto_link}} to specify PyDec to use your function:
```python
pydec.decOVF.set_decomposition_func("your_algorithm_name")
...

with pydec.decOVF.using_decomposition_func("your_algorithm_name"):
    ...
```
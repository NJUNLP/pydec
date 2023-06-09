# Extending PyDec

## Extending the decomposition algorithm

If you only need to modify the decomposition algorithm of the activation function, see [Customizing the decomposition algorithm](decompose-activation-functions.md#customizing-the-decomposition-algorithm).

## Overriding PyDec APIs

If you need to override PyDec APIs instead of just modifying the decomposition algorithm. You can use {{#auto_link}}pydec.overrides.register_torch_function{{/auto_link}} to register your customized functions so that when you invoke the specified torch API, the invocation is dispatched to the customized function instead of the PyDec API.
```python
@pydec.overrides.register_torch_function(torch.add)
def my_add(input: pydec.Composition, other: Any) -> pydec.Composition:
    print("invoking my_add!")
    return input + len(other)

c = pydec.zeros(3, c_num=2)

print(torch.add(c, c))
```

Out:
```python
"""
invoking my_add!
composition{
  components:
    tensor([0., 0., 0.]),
    tensor([0., 0., 0.]),
  residual:
    tensor([3., 3., 3.])}
"""
```

Note that this only affects invocations via the torch API, so you can use the PyDec API in your customized functions without worrying about infinite recursion.
```python
print(pydec.add(c, c)) # not invoke `my_add`
```

Out:
```python
"""
composition{
  components:
    tensor([0., 0., 0.]),
    tensor([0., 0., 0.]),
  residual:
    tensor([0., 0., 0.])}
"""
```
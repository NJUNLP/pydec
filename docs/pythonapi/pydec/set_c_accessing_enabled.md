# PYDEC.SET_C_ACCESSING_ENABLED
> CLASS pydec.set_c_accessing_enabled(mode)

Context-manager that sets indexing and slicing of components on or off. If indexing and slicing of components is enabled, the first dimension of indices is used to access components. If indexing a single component, a tensor is returned. If indexing multiple components, they are returned as a composition. Also, `len()` returns the number of components and `iter()` returns an iterator that traverses each component.

Also functions as a decorator.

**Parameters:**

* **mode** (*{{{python_bool}}}*) –  Flag whether to enable component accessing (*True*), or disable (*False*).

Example:
```python
print(pydec.is_c_accessing_enabled())
with pydec.set_c_accessing_enabled(True):
    print(pydec.is_c_accessing_enabled())
    with pydec.set_c_accessing_enabled(False):
        print(pydec.is_c_accessing_enabled())
```

Out:
```python
"""
False
True
False
"""
```
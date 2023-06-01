# PYDEC.NUMEL
> pydec.numel(int) →  {{python_int}}


Returns the total number of elements in the recovery of the `input` composition.

?> To get the total number of elements of all components, use {{#auto_link}}pydec.c_numel{{/auto_link}}.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) - the input composition.

Example:
```python
>>> a = pydec.zeros((1, 2, 3, 4, 5), component_num=4)
>>> pydec.numel(a)
"""
120
"""
>>> t = torch.zeros(3, 4, 4)
>>> a = pydec.Composition(t)
>>> pydec.numel(a)
"""
16
"""
```
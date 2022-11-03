---
title: "numel"
description: pydec.numel
---
# PYDEC.NUMEL
{% include function.html content="pydec.numel(input) -> int" %}

Returns the total number of elements in the individual components of the `input` composition.

{% include function.html content="Parameters:" %}

* **input** ({% include codelink.html name="Composition" path="pythonapi/pydec.Composition" %}) - the input composition.

Example:
```
>>> a = pydec.Composition((1, 2, 3, 4, 5), component_num=4)
>>> pydec.numel(a)
120
>>> t = torch.zeros(3, 4, 4)
>>> a = pydec.Composition(t)
>>> pydec.numel(a)
16
```


{% include class.html content="pydec.Composition" %}

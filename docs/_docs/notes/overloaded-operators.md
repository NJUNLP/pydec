---
title: "Overloaded Operators"
description: Overloaded Operators
---

# Overloaded Operators

Similar to pytorch, we have overloaded some operators to fit the code style of PyTorch developers.

Operators overloaded by PyDec:

| Operator | API           | Description                      |
| -------- | ------------- | -------------------------------- |
| []       | `__getitem__` | Evaluation of `self[key]`        |
| []       | `__setitem__` | Assignment to `self[key]`        |
| +        | `__pos__`     | The unary arithmetic operations  |
| -        | `__neg__`     | The unary arithmetic operations  |
| +        | `__add__`     | The binary arithmetic operations |
| -        | `__sub__`     | The binary arithmetic operations |
| @        | `__matmul__`  | The binary arithmetic operations |
| *        | `__mul__`     | The binary arithmetic operations |
| /        | `__truediv__` | The binary arithmetic operations |
| ==       | `__eq__`      | The "rich comparison" methods    |
| >        | `__gt__`      | The "rich comparison" methods    |
| <        | `__lt__`      | The "rich comparison" methods    |
| >=       | `__ge__`      | The "rich comparison" methods    |
| <=       | `__le__`      | The "rich comparison" methods    |

The behavior of these operators is similar to the definition of pytoch, but some of them have been added with additional functionality. Refer to our API documentation (TODO) for detailed definitions.
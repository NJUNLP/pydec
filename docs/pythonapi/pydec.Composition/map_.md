# PYDEC.COMPOSITION.MAP_
> Composition.map_(composition, callable) →  {{{pydec_Composition}}}

Applies `callable` for each element of components in `self` composition and the given composition and stores the results in `self` composition.  self composition and the given composition must be [broadcastable](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics).

The `callable` should have the signature:
```python
def callable(a, b) -> number
```
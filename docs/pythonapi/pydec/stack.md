# PYDEC.STACK
> pydec.stack(compositions, dim=0, *, out=None) →  {{{pydec_Composition}}}

Concatenates a sequence of compositions along a new dimension.

All compositions need to be of the same size.

**Parameters:**

* **compositions** (*sequence of Compositions*) – sequence of compositions to concatenate.
* **dim** (*{{{python_int}}}*) - dimension to insert. Has to be between 0 and the number of dimensions of concatenated tensors (inclusive).


**Keyword Arguments:**
* **out** (*{{{pydec_Composition}}}, optional*) - the output composition.
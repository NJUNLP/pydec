# PYDEC.SELECT
> pydec.select(input, dim, index) →  {{{pydec_Composition}}}

Slices the `input` composition along the selected dimension at the given index. This function returns a view of the original composition with the given dimension removed.

**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition.
* **dim** (*{{{python_int}}}*) – the dimension to slice.
* **dim** (*{{{python_int}}}*) – the index to select with.

?> {{#auto_link}}pydec.select short{{/auto_link}} is equivalent to slicing. For example, `composition.select(0, index)` is equivalent to `composition[index]` and `composition.select(2, index)` is equivalent to `composition[:,:,index]`.
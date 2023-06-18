# PYDEC.COMPOSITION.TYPE_AS
> Composition.type_as(other) → {{{python_str}}} or {{{pydec_Composition}}}

Returns this composition cast to the type of the given tensor or composition.

This is a no-op if the composition is already of the correct type. This is equivalent to `self.type(other.type())`

**Parameters:**

* **other** (*{{{torch_Tensor}}} or {{{pydec_Composition}}}*) - the tensor or composition which has the desired type.
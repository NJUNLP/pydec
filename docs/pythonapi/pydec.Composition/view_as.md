# PYDEC.COMPOSITION.VIEW_AS
> Composition.view_as(*shape) → {{{pydec_Composition}}}

View this composition as the same shape as `other`. `self.view_as(other)` is equivalent to `self.view(other.size())`.

Please see {{#auto_link}}pydec.Composition.view short:2{{/auto_link}} for more information about `view`.

?> `Composition.view_as()` does not change the shape of the composition dimension.

**Parameters:**

* **other** (*{{{torch_Tensor}}} or {{{pydec_Composition}}}*) - the result composition has the same size as `other`.
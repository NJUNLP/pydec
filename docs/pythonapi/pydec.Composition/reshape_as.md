# PYDEC.COMPOSITION.RESHAPE_AS
> Composition.reshape_as(other) →  {{{pydec_Composition}}}

Returns this composition as the same shape as `other`. `self.reshape_as(other)` is equivalent to `self.reshape(other.size())`. This method returns a view if `other.size()` is compatible with the current shape. See {{#auto_link}}pydec.Composition.view{{/auto_link}} on when it is possible to return a view.

Please see {{#auto_link}}pydec.reshape{{/auto_link}} for more information about `reshape`.

**Parameters:**

* **other** (*{{{pydec_Composition}}}*) - The result composition has the same shape as `other`.
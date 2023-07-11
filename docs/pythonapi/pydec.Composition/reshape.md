# PYDEC.COMPOSITION.RESHAPE
> Composition.reshape(*shape) →  {{{pydec_Composition}}}

Returns a composition with the same data and number of elements as `self` but with the specified shape.  This method returns a view if `shape` is compatible with the current shape. See {{#auto_link}}pydec.Composition.view{{/auto_link}} on when it is possible to return a view.

See {{#auto_link}}pydec.reshape{{/auto_link}}.

**Parameters:**

* **shape** (*tuple of int*) - the desired shape.
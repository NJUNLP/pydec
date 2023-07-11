# PYDEC.COMPOSITION.CONTIGUOUS
> Composition.contiguous(memory_format=torch.contiguous_format) →  {{{pydec_Composition}}}

Returns a contiguous in memory composition containing the same data as `self` composition.  If `self` composition is already in the specified memory format, this function returns the `self` composition.

**Parameters:**

* **memory_format** (*{{{torch_memory_format}}}, optional*) – the desired memory format of returned composition. Default: **torch.preserve_format**.
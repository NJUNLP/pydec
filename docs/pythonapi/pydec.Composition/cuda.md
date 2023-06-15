# PYDEC.COMPOSITION.CUDA
> Composition.cuda(device=None, non_blocking=False) →  {{{pydec_Composition}}}

Returns a copy of this object in CUDA memory.

If this object is already in CUDA memory and on the correct device, then no copy is performed and the original object is returned.


**Parameters:**

* **device** (*{{{torch_device}}}*) – The destination GPU device. Defaults to the current CUDA device.
* **non_blocking** (*{{{python_bool}}}*) - If *True* and the source is in pinned memory, the copy will be asynchronous with respect to the host. Otherwise, the argument has no effect. Default: *False*.
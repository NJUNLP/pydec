# PYDEC.NN.FUNCTIONAL.EMBEDDING
> pydec.nn.functional.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False) →  {{pydec_Composition}}

A simple lookup table that looks up embeddings in a fixed dictionary and size.

This is a module that accepts a Composition initialized by index, often used to retrieve word embeddings using indices. The input to the module is a {{{pydec_IndexComposition}}}, and the embedding matrix, and the output is the corresponding {{pydec_Composition}} of word embeddings.

See [torch.nn.Embedding()](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding) for details about `embedding`.

**Parameters:**

* **input** (*{{{pydec_IndexComposition}}}*) – Composition containing indices into the embedding matrix.
* **weight** (*{{{torch_Tensor}}}*) - The embedding matrix with number of rows equal to the maximum possible index + 1, and number of columns equal to the embedding size.
* **padding_idx** (*{{{python_int}}}, optional*) - If specified, the entries at `padding_idx` do not contribute to the gradient; therefore, the embedding vector at `padding_idx` is not updated during training, i.e. it remains as a fixed "pad".
* **max_norm** (*{{{python_float}}}, optional*) - If given, each embedding vector with norm larger than `max_norm` is renormalized to have norm `max_norm`. Note: this will modify `weight` in-place.
* **norm_type** (*{{{python_float}}}, optional*) - The p of the p-norm to compute for the `max_norm` option. Default **2**.
* **scale_grad_by_freq** (*{{{python_bool}}}, optional*) - If given, this will scale gradients by the inverse of frequency of the words in the mini-batch. Default **False**.
* **sparse** (*{{{python_bool}}}, optional*) - If **True**, gradient w.r.t. `weight` will be a sparse tensor. See Notes under [torch.nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding) for more details regarding sparse gradients.
  

**Return type:**
*{{{pydec_Composition}}}*
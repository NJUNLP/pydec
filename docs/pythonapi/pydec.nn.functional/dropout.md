# PYDEC.NN.FUNCTIONAL.DROPOUT
> pydec.nn.functional.dropout(input, p=0.5, training=True, inplace=False) →  {{pydec_Composition}}

During training, randomly zeroes some of the elements of the input composition with probability `p` using samples from a Bernoulli distribution.

The dropout between components are synchronized. That is, if an element is dropped, then the corresponding elements in all components are dropped.

See [torch.nn.Dropout()](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout) for details about `dropout`.
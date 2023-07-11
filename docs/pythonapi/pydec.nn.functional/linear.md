# PYDEC.NN.FUNCTIONAL.LINEAR
> pydec.nn.functional.linear(input, weight, bias=None) →  {{pydec_Composition}}

Applies a linear transformation to the incoming data: $y=xA^T+b$. Where the linear transformation is applied to each component, but the bias is only added to the residual:

$$
y_i=x_iA^T\\
y^R=x^RA^T+b
$$


**Parameters:**

* **input** (*{{{pydec_Composition}}}*) – the input composition with shape $(∗,in\_features)$ where * means any number of additional dimensions, including none.
* **weight** (*{{{torch_Tensor}}}*) - the weight tensor with shape $(out\_features,in\_features)$ or $(in\_features)$.
* **bias** (*{{{torch_Tensor}}}*) - the bias tensor with shape $(out\_features)$ or $()$.

**Return type:**
*{{{pydec_Composition}}}*
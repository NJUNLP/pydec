# PYDEC.NN.FUNCTIONAL

## Convolution functions
| API                                                                                 | Description                                                                    |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| {{#auto_link}}pydec.nn.functional.conv2d short with_parentheses:false{{/auto_link}} | Applies a 2D convolution over an input image composed of several input planes. |

## Pooling functions
| API                                                                                     | Description                                                                     |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| {{#auto_link}}pydec.nn.functional.max_pool2d short with_parentheses:false{{/auto_link}} | Applies a 2D max pooling over an input signal composed of several input planes. |

## Non-linear activation functions
| API                                                                                      | Description                                                                                    |
| ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| {{#auto_link}}pydec.nn.functional.relu short with_parentheses:false{{/auto_link}}        | Applies the rectified linear unit function element-wise.                                       |
| {{#auto_link}}pydec.nn.functional.relu_ short with_parentheses:false{{/auto_link}}       | In-place version of {{#auto_link}}pydec.nn.functional.relu short{{/auto_link}}.                |
| {{#auto_link}}pydec.nn.functional.leaky_relu short with_parentheses:false{{/auto_link}}  | Applies element-wise, $\text{LeakyReLU}(x)=\max (0,x) + \text{negative\_slope} * \min (0, x)$. |
| {{#auto_link}}pydec.nn.functional.leaky_relu_ short with_parentheses:false{{/auto_link}} | In-place version of {{#auto_link}}pydec.nn.functional.leaky_relu_ short{{/auto_link}}.         |
| {{#auto_link}}pydec.nn.functional.softmax short with_parentheses:false{{/auto_link}}     | Applies a softmax function.                                                                    |
| {{#auto_link}}pydec.nn.functional.tanh short with_parentheses:false{{/auto_link}}        | Applies element-wise, $\text{Tanh}(x)=\tanh (x)=\frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}$.    |
| {{#auto_link}}pydec.nn.functional.sigmoid short with_parentheses:false{{/auto_link}}     | Applies the element-wise function $\text{Sigmoid}(x)=\frac{1}{1+\exp(-x)}$.                    |
| {{#auto_link}}pydec.nn.functional.layer_norm short with_parentheses:false{{/auto_link}}  | Applies Layer Normalization for last certain number of dimensions.                             |

## Linear functions
| API                                                                                 | Description                                                       |
| ----------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| {{#auto_link}}pydec.nn.functional.linear short with_parentheses:false{{/auto_link}} | Applies a linear transformation to the incoming data: $y=xA^T+b$. |

## Dropout functions
| API                                                                                  | Description                                                                                                                                      |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| {{#auto_link}}pydec.nn.functional.dropout short with_parentheses:false{{/auto_link}} | During training, randomly zeroes some of the elements of the input composition with probability `p` using samples from a Bernoulli distribution. |

## Sparse functions
| API                                                                                    | Description                                                                    |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| {{#auto_link}}pydec.nn.functional.embedding short with_parentheses:false{{/auto_link}} | A simple lookup table that looks up embeddings in a fixed dictionary and size. |

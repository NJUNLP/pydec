# Reduce Memory Overhead

## Space Complexity
The space overhead of PyDec is $n$ times the original (excluding model parameters), where $n$ is the number of components. If decomposition is performed at the token level, the space overhead grows linearly with the sentence length; if decomposition is performed at the hidden dimension, the space overhead is independent of the sentence length, but the hidden dimension is usually larger than the sentence length. Therefore, the management of memory is important.

## Recommendations for reducing memory overhead

### Use half-precision calculations
If you will not be particularly demanding in terms of precision, using half-precision can help you save memory. The errors introduced by using half-precision can be found [here](error-control.md#error-statistics).

### Time-space trade-off
The number of components does not have to be equal to the number of tokens (or some other dimension). Let's say the input has two sentences, and you only care about the overall contribution of each sentence. You can specify a number of components of 2 right at the initialization:
```python
sentence_len1 = 24
sentence_len2 = 17
hidden_dim = 728

input = torch.randn(sentence_len1 + sentence_len2, hidden_dim)
c_input = pydec.zeros(sentence_len1 + sentence_len2, hidden_dim, c_num=2)

# initialization
c_input()[0, :sentence_len1] = input[:sentence_len1]
c_input()[1, sentence_len1:] = input[sentence_len1:]
```

Anyway, you still want to know the contribution of each token. No problem, this time you still use only 2 components, but the whole input is divided into the token 0 and other:
```python
c_input = pydec.zeros(sentence_len1 + sentence_len2, hidden_dim, c_num=2)

# initialization
c_input()[0, 0] = input[0]
c_input()[1, 1:] = input[1:]
```
From the output of the model, you can get the contribution of token 0. If the decomposition algorithm is consistent, this contribution is the same as the one obtained by evaluating all tokens simultaneously. Then, you do the same operation once for each token and you get the whole answer. Of course, if you have enough memory, you can evaluate more than one token at a time. It's flexible.
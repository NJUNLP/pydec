# Variable Tracing

## What is tracing

In order to obtain the decomposition of hidden states in the network, PyDec is based on the following process: starting from the initialized composition, perform the same operators applied to the model inputs in the same order. We call this process **tracing**. An example of a pseudocode:
```python
class Model(torch.nn.Module):
    def __init__(self) -> None:
        ...

    def forward(self, input):
        x1 = self.linear1(input)
        x2 = torch.nn.functional.relu(x1)
        out = self.linear2(x2)

        # perform tracing

        # initialize composition
        c_input = pydec.zeros(input.size(), c_num=input.size(-1))
        c_input = pydec.diagonal_init(c_input, input, dim=-1)

        # perform the same operators in the same order
        c1 = self.linear1(input)
        c2 = torch.nn.functional.relu(c1)
        c_out = self.linear2(c2)

        return out, c_out
```

## Tracing failures
Tracing is used to guarantee that each composition corresponds to a tensor in the original forward. In the above example, `c_input`, `c1`, `c2` and `c_out` correspond to `input`, `x1`, `x2` and `out`, respectively. So each composition is a decomposition of its corresponding tensor and we have:

```python
c_input.recovery == input # true
c1.recovery == x1 # true
c2.recovery == x2 # true
c_out.recovery == out # true
```

However, once the recovery of a composition is not equal to its corresponding tensor (ground truth), the current as well as the subsequent tracing fails. PyDec guarantees as much as possible that the provided operators can perform the tracing successfully.

Some operators guarantee the success of the tracing in any case, e.g., {{#auto_link}}pydec.add{{/auto_link}}, {{#auto_link}}pydec.mul{{/auto_link}} and {{#auto_link}}pydec.mean{{/auto_link}}.

Some operators (almost) always cause tracing failures: e.g., {{#auto_link}}pydec.round{{/auto_link}}.

Some other operators depends:
```python
c = pydec.masked_fill(c, c > 0, 1.0) # tracing failure

c = pydec.masked_fill(c, c > 0, 0.0) # tracing success
```

## Avoid tracing failures

To check if the tracing fails, you should first check if the tracing to the model output is correct, if so then all tracing for intermediate variables should be correct.
```python
out, c_out = model(input)
assert torch.all((c_out.recovery - out).abs() < 1e-6)
```

If the tracing fails, you need to check the operators applied to the input one by one in the forward and then fix the failed operations. If there is no way to fix it, and you wish to ignore this failure and continue the tracing. Consider combining the error of the composition with the ground truth into the residual before you can further check the subsequent operations:
```python
...
c2 = ops(c1)
ref = ops(c1.recovery) # the ground truth

torch.all((c2.recovery - ref).abs() < 1e-6) # false

c2.residual[:] += ref - c2.recovery

torch.all((c2.recovery - ref).abs() < 1e-6) # true
...
```
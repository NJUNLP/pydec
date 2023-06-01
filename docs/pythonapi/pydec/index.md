# PYDEC
The pydec package contains data structures for compositions and defines mathematical operations over these compositions. Additionally, it provides many useful utilities. 

## Compositions

| API                                                                    | Description                                                                                   |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| {{#auto_link}}pydec.numel short with_parentheses:false{{/auto_link}}   | Returns the total number of elements in the individual components of the `input` composition. |
| {{#auto_link}}pydec.c_numel short with_parentheses:false{{/auto_link}} | Returns the total number of elements in all components of the `input` composition.            |
| {{#auto_link}}pydec.numc short with_parentheses:false{{/auto_link}}    | Returns the number of components in the `input` composition.                                  |

## Creation Ops
?> To create Composition by class constructor, use {{#auto_link}}pydec.Composition{{/auto_link}}.

| API                                                                       | Description                                                                                                                                                                         |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| {{#auto_link}}pydec.void short with_parentheses:false{{/auto_link}}       | Returns an void composition.                                                                                                                                                        |
| {{#auto_link}}pydec.zeros short with_parentheses:false{{/auto_link}}      | Returns a composition whose components filled with the scalar value *0*, with the shape and the component number defined by the variable argument `size` and `c_num`, respectively. |
| {{#auto_link}}pydec.zeros_like short with_parentheses:false{{/auto_link}} | Returns a composition whose components filled with the scalar value *0*, with the same shape as `input`.                                                                            |

## Indexing, Slicing, Joining, Mutating Ops

| API                                                                             | Description                                                                                                                                                  |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| {{#auto_link}}pydec.cat short with_parentheses:false{{/auto_link}}              | Concatenates the given sequence of `seq` compositions in the given dimension.                                                                                |
| {{#auto_link}}pydec.concat short with_parentheses:false{{/auto_link}}           | Alias of {{#auto_link}}pydec.cat{{/auto_link}}.                                                                                                              |
| {{#auto_link}}pydec.concatenate short with_parentheses:false{{/auto_link}}      | Alias of {{#auto_link}}pydec.cat{{/auto_link}}.                                                                                                              |
| {{#auto_link}}pydec.c_cat short with_parentheses:false{{/auto_link}}            | Concatenates the given sequence of `seq` compositions in the component dimension.                                                                            |
| {{#auto_link}}pydec.gather short with_parentheses:false{{/auto_link}}           | Gathers values along an axis specified by *dim*.                                                                                                             |
| {{#auto_link}}pydec.index_select short with_parentheses:false{{/auto_link}}     | Returns a new composition which indexes the `input` composition along dimension `dim` using the entries in `index` which is a *LongTensor*.                  |
| {{#auto_link}}pydec.c_index_select short with_parentheses:false{{/auto_link}}   | Returns a new composition which indexes the `input` composition along the component dimension using the entries in `index` which is a *LongTensor*.          |
| {{#auto_link}}pydec.masked_select short with_parentheses:false{{/auto_link}}    | Returns a new 1-D composition which indexes the `input` composition according to the boolean mask `mask` which is a *BoolTensor*.                            |
| {{#auto_link}}pydec.masked_fill short with_parentheses:false{{/auto_link}}      | Fills elements of each component in `input` composition with `value` where `mask` is *True*.                                                                 |
| {{#auto_link}}pydec.c_masked_fill short with_parentheses:false{{/auto_link}}    | Fills components of the `input` composition with `value` where `mask` is *True*.                                                                             |
| {{#auto_link}}pydec.permute short with_parentheses:false{{/auto_link}}          | Returns a view of the original composition `input` with its components' dimensions permuted.                                                                 |
| {{#auto_link}}pydec.reshape short with_parentheses:false{{/auto_link}}          | Returns a composition with the same data and number of elements as `input`, but with the specified shape.                                                    |
| {{#auto_link}}pydec.scatter short with_parentheses:false{{/auto_link}}          | Out-of-place version of {{#auto_link}}pydec.Composition.scatter_{{/auto_link}}.                                                                              |
| {{#auto_link}}pydec.diagonal_scatter short with_parentheses:false{{/auto_link}} | Embeds the values of the `src` tensor into `input` composition along the diagonal elements of every component of `input`, with respect to `dim1` and `dim2`. |
| {{#auto_link}}pydec.diagonal_init short with_parentheses:false{{/auto_link}}    | Embeds the values of the `src` tensor into `input` composition along the diagonal components of `input`, with respect to `dim`.                              |
| {{#auto_link}}pydec.squeeze short with_parentheses:false{{/auto_link}}          | Returns a composition with all the dimensions of components of `input` of size *1* removed.                                                                  |
| {{#auto_link}}pydec.stack short with_parentheses:false{{/auto_link}}            | Concatenates every component of a sequence of compositions along a new dimension.                                                                            |
| {{#auto_link}}pydec.transpose short with_parentheses:false{{/auto_link}}        | Returns a composition that is a transposed version of `input`.                                                                                               |
| {{#auto_link}}pydec.unsqueeze short with_parentheses:false{{/auto_link}}        | Returns a new composition with a dimension of size one inserted at the specified position of each component.                                                 |

## Math operations
### Pointwise Ops

| API                                                                     | Description                                                                                           |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| {{#auto_link}}pydec.add short with_parentheses:false{{/auto_link}}      | Adds `other`, scaled by `alpha`, to `input`.                                                          |
| {{#auto_link}}pydec.div short with_parentheses:false{{/auto_link}}      | Divides each element of every component of the `input` input by the corresponding element of `other`. |
| {{#auto_link}}pydec.divide short with_parentheses:false{{/auto_link}}   | Alias for {{#auto_link}}pydec.div{{/auto_link}}.                                                      |
| {{#auto_link}}pydec.mul short with_parentheses:false{{/auto_link}}      | Multiplies each component of `input` by `other`.                                                      |
| {{#auto_link}}pydec.multiply short with_parentheses:false{{/auto_link}} | Alias for {{#auto_link}}pydec.mul{{/auto_link}}.                                                      |
| {{#auto_link}}pydec.round short with_parentheses:false{{/auto_link}}    | Rounds elements of each component of `input` to the nearest integer.                                  |
| {{#auto_link}}pydec.sub short with_parentheses:false{{/auto_link}}      | Subtracts `other`, scaled by `alpha`, from `input`.                                                   |
| {{#auto_link}}pydec.subtract short with_parentheses:false{{/auto_link}} | Alias for {{#auto_link}}pydec.sub{{/auto_link}}.                                                      |


### Reduction Ops

| API                                                                  | Description                                                                          |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| {{#auto_link}}pydec.all short with_parentheses:false{{/auto_link}}   | Tests if all elements in `input.recovery` evaluate to True.                          |
| {{#auto_link}}pydec.any short with_parentheses:false{{/auto_link}}   | Tests if any element in `input.recovery` evaluates to True.                          |
| {{#auto_link}}pydec.mean short with_parentheses:false{{/auto_link}}  | Returns the mean value of all elements in each component of the `input` composition. |
| {{#auto_link}}pydec.sum short with_parentheses:false{{/auto_link}}   | Returns the sum of all elements in each component of the `input` composition.        |
| {{#auto_link}}pydec.c_sum short with_parentheses:false{{/auto_link}} | Returns the sum of all components of the `input` composition.                        |


## Other Operations

Currently PyDec does not yet fully cover the API supported by PyTorch. As a workaround, PyDec provides the function {{#auto_link}}pydec.c_apply{{/auto_link}} and {{#auto_link}}pydec.c_map{{/auto_link}}, which will call the specified PyTorch function and pass in the combined component tensors as arguments. This usually works on [Pointwise Ops](https://pytorch.org/docs/stable/torch.html#pointwise-ops), while errors may occur on other operations.

| API                                                                    | Description                                                                 |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| {{#auto_link}}pydec.c_apply short with_parentheses:false{{/auto_link}} | Applies the function `callable` to each component of `input`.               |
| {{#auto_link}}pydec.c_map short with_parentheses:false{{/auto_link}}   | Applies `callable` for each component of `input` and the given composition. |
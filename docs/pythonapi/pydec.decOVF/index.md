# PYDEC.DECOVF

{{#auto_link}}pydec.decOVF with_parentheses:false{{/auto_link}} is a module for decomposing nonlinear one variable functions (OVF), which provides implementations of some decomposition algorithms. In addition, it provide interfaces to register user-defined algorithms.

## Coverage
{{#auto_link}}pydec.decOVF with_parentheses:false{{/auto_link}} should cover all nonlinear one variable functions, but linear functions are out of scope. That is, all invocations to nonlinear OVFs should be dispatched here. Common nonlinear OVFs include nonlinear activation functions, as well as some element-wise operators ([torch.reciprocal()](https://pytorch.org/docs/stable/generated/torch.reciprocal.html#torch-reciprocal), etc.).

## Configuring the decomposition algorithm

| API                                                                                               | Description                                                                |
| ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| {{#auto_link}}pydec.decOVF.get_decomposition_name short with_parentheses:false{{/auto_link}}      | Returns the name of the currently enabled decomposition algorithm.         |
| {{#auto_link}}pydec.decOVF.get_decomposition_func short with_parentheses:false{{/auto_link}}      | Returns the function of the currently enabled decomposition algorithm.     |
| {{#auto_link}}pydec.decOVF.get_decomposition_args short with_parentheses:false{{/auto_link}}      | Returns the currently configured arguments of the decomposition algorithm. |
| {{#auto_link}}pydec.decOVF.register_decomposition_func short with_parentheses:false{{/auto_link}} | Decorator for registering a decomposition algorithm with `name`.           |
| {{#auto_link}}pydec.decOVF.set_decomposition_func short with_parentheses:false{{/auto_link}}      | Specify the decomposition algorithm with `name` globally.                  |
| {{#auto_link}}pydec.decOVF.set_decomposition_args short with_parentheses:false{{/auto_link}}      | Specify the arguments of the decomposition algorithm globally.             |
| {{#auto_link}}pydec.decOVF.using_decomposition_func short with_parentheses:false{{/auto_link}}    | Context-manager that specify the decomposition algorithm with `name`.      |
| {{#auto_link}}pydec.decOVF.using_decomposition_args short with_parentheses:false{{/auto_link}}    | Context-manager that specify the arguments of the decomposition algorithm. |

## Decomposition algorithms
| Name          | API                                                                                               | Description                                                                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| affine        | {{#auto_link}}pydec.decOVF.affine_decomposition short with_parentheses:false{{/auto_link}}        | Mapping `input` to output of OVF based on affine transformation.                                                                            |
| scaling       | {{#auto_link}}pydec.decOVF.scaling_decomposition short with_parentheses:false{{/auto_link}}       | Mapping `input` to output of OVF based on scaling.                                                                                          |
| abs_affine    | {{#auto_link}}pydec.decOVF.abs_affine_decomposition short with_parentheses:false{{/auto_link}}    | Mapping `input` to output of OVF based on affine transformation. All components in the `input` are first taken in absolute value.           |
| hybrid_affine | {{#auto_link}}pydec.decOVF.hybrid_affine_decomposition short with_parentheses:false{{/auto_link}} | Mapping `input` to output of OVF based on affine transformation. Use `threshold` to control whether to take absolute values for components. |
| none          | {{#auto_link}}pydec.decOVF._none_decomposition short with_parentheses:false{{/auto_link}}         | No decomposition is performed. All components are zeroed and the output of OVF is assigned to the residual.                                 |

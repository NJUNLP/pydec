# Changelog

This is a manually generated log to track changes to the repository for each release. 
Each section should include general headers such as **Implemented enhancements** 
and **Merged pull requests**. All closed issued and bug fixes should be 
represented by the pull requests that fixed them.
Critical items to know are:

 - renamed commands
 - deprecated / removed commands
 - changed defaults
 - backward incompatible changes
 - migration guidance
 - changed behaviour

## PyDec 0.2.0
* Add non linear decompose algorithm.
* Add nn modules (Linear, Containers, ReLU).
* Add `conv2d()`, `relu()`, `leaky_relu()`, `tanh()`, `sigmoid()` and `max_pool2d()`.
* Add new feature: autotracing.
* Add fast algorithm for `legacy_relu()`.
* Add `abs()`, `__rmatmul__()`, `mv()`, `mm()` function.
* Add `apply_()` and `map_()`.
* Add `c_apply()` and `c_map()`.
* Add `c_stack()`.
* Add argument `sum_residual` for `c_cat()`.
* Add `zeros()` and support for torch versions < 1.11
* Add customized api registration.
* Update docs.
* Fix bugs.
* 
## PyDec 0.1.1
* Update doc notes and api documentation.
* Improve doc site style.
* Add decorator feature to bias decomposition context managers.
* Simplify bias decomposition algorithm names.
* Make `Composition.mul()` support `out` argument.
* Add `Composition.div()` and `Composition.div_`().
* Add `is_error_checking_enabled()`.
* Add alias `concat()` and `concatenate()`.
* Add alias `subtract()` and `multiply()`.
* Add `count_residual` argument for `c_numel()`.
* Add `div()` and `divide()`.
* Fix bugs.
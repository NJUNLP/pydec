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
* Add `count_residual()` argument for `c_numel()`.
* Add `div()` and `divide()`.
* Fix bugs.
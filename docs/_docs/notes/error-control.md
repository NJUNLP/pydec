---
title: "Error Control"
tags: 
 - "error control"
description: Error Control
---
# Error Control

Although in theory the recovery from Composition is exactly equivalent to the ground truth. However, in practice, there will be errors brought by the computer. Especially in deep networks, the error may be magnified to an unacceptable degree. We give some suggestions for reducing errors and provide tools for error checking.

## Error reduction

Our most recommended method for reducing errors is to use double precision computations, usually by simply adding `model=model.double()` after the model is loaded. If you enable double precision calculations, the error from the decomposition is almost negligible.

When double precision computation cannot be enabled for speed and memory reasons, you may consider adding the error term to Composition as bias. Depending on the bias reallocation policy, the error is added to the residual or assigned to each component.

You can even consider making the ground truth equal to Composition's recovery, but this may change the classification result of the network.

## Error checking
You can use `PyDec.check_error` to check the error of the given Composition and reference. In order to provide ground truth as a reference, you usually need to keep the forward process of the original network. We recommend that you use it often during development, not only for error control, but also to help you find bugs in your code.

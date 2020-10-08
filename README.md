# mTDRdemo
Self-contained code to demonstrate how to use mTDR. An analysis framework for of trial-structured neural population data using a combination of regression and dimensionality reduction.

This MATLAB code is a reference implementation for the analyses found
in [Aoi, Mante, and Pillow 2020](https://www.nature.com/articles/s41593-020-0696-5).


Downloading the repository
------------

- **From command line:**

     ```git clone git@github.com:pillowlab/mTDRdemo.git```

- **In browser:**   click to
  [Download ZIP](https://github.com/pillowlab/mTDRdemo/archive/master.zip)
  and then unzip archive


Example Script
-
Open ``mTDRdemo.m`` to see how we estimate the parameters and the number of dimensions by AIC. ``demoLearning.m`` is a more step-by-step view of the functions used for parameter learning when the number of dimensions have been specified.


Simple Overview
-------------

Suppose we record spike responses from a single neuron during a
complex behavioral experiment, and would like to know what aspects of
the stimulus or behavior are encoded in the neural response. This code
package allows us to discover such dependencies using Poisson GLM
regression.

Consider a simple example in which a neuron encodes two experimental
variables: the time at which a visual target appears, and the motion
strength of a moving-dots stimulus. The regressors are
the time at which the targets appear, and the time,
duration, and strength ("coherence") of the moving dots on each
trial.   


## Reference

- MC Aoi, V Mante, &  JW Pillow
 (2020).
 [Prefrontal cortex exhibits multidimensional dynamic encoding during decision-making](https://doi.org/10.1038/s41593-020-0696-5) Nature Neuroscience 2020.

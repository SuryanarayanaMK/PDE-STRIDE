# PDE-STRIDE
This repository contains codes and data-sets for the PDE inference from limited spatio-temporal data.

Code for the paper "Stability selection enables robust learning of partial differential equations from limited noisy data"
by Suryanarayana Maddu, Bevan L. Cheeseman, Ivo F. Sbalzarini and Chrisitan L. Mueller.

Files description

*********************************  Iterative_solvers_l0_l1.py  ******************************************

This files contains the following functions: 

build_linear_system(--) : Outputs Theta and Ut for gridded values of the state-variable measurements.
Iterative_hard_thresholding_debias(--) : Outputs IHT-d sparse solutions
STridge(--) : Outputs STRidge sparse solutions
stability_selection(--) : Outputs statistics aggregated over sub-sampled based repetitons of sparse-regression

**********************************************************************************************************

****************************************   basic_demo.ipynb   ********************************************
Used to demonstrate basic concepts like regularization paths and stability plots using Burgers data-set.

The notebook produces stabilty plots for LASSO, STRidge and IHT-d sparsity-promoting algorithms.

Run the notebook for inferring Burgers equations from limited and noisy-data set.

Parameters to play-around: N = sample size
                           D/P - to set the dictionary size p. D - max derivatives order, P - max polynomial order 
                           noise-level
***********************************************************************************************************

# PDE-STRIDE
This repository contains codes and data-sets for PDE inference from limited spatio-temporal data.

Code for the paper "Stability selection enables robust learning of partial differential equations from limited noisy data"
by Suryanarayana Maddu, Bevan L. Cheeseman, Ivo F. Sbalzarini and Chrisitan L. Mueller.

## File descriptions:

*********************************  Iterative_solvers_l0_l1.py  ******************************************

These files contain the following functions: 

build_linear_system(--) : Outputs Theta and Ut for gridded values of the state-variable measurements.
Iterative_hard_thresholding_debias(--) : Outputs IHT-d sparse solutions
STridge(--) : Outputs STRidge sparse solutions
stability_selection(--) : Outputs statistics aggregated over sub-sampled based repetitons of sparse-regression

**********************************************************************************************************

****************************************   basic_demo.ipynb   ********************************************
Used to demonstrate basic concepts like regularization paths and stability plots using Burgers data-set.

The notebook produces stabilty plots for LASSO, STRidge and IHT-d sparsity-promoting algorithms.

Run the notebook for inferring Burgers equations from a limited and noisy-data set.

Parameters to play-around: N = sample size
                           D/P - to set the dictionary size p. D - max derivatives order, P - max polynomial order 
                           noise-level
***********************************************************************************************************

********************************************   Burgers.py   ***********************************************
 
Code to reproduce the plots for Burgers recovery upto 4-5 noise-levels for N = 250. 

De-noising is achieved through truncating SVD singular values at the elbow. 
 
Writes the output to files "IHTd_Burgers_n + str(sigma) + _N+str(N)_p+str(p)_dim+str(SVDmode)_B250.txt"

The output files are read in the notebook "Burgers_results_visualization.ipynb" for visualization of stability plots.

Try to play around with the sample-size (N), dictionary-size (p) and noise-levels.

***********************************************************************************************************
 

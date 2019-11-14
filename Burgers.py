#pylab.rcParams['figure.figsize'] = (12, 8)
import matplotlib
matplotlib.use('Agg')
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from Iterative_solvers_l0_l1 import *
import scipy.io as sio
import itertools
import pandas as pd

print(" reading data .... ")

X_APR    = np.loadtxt('Burgers/X_burgers.dat', dtype='double')
T_APR    = np.loadtxt('Burgers/T_burgers.dat', dtype='double') 

data = sio.loadmat('Burgers/burgers.mat')
u = np.real(data['usol'])
dx = X_APR[1] - X_APR[0]
dt = T_APR[1] - T_APR[0]
print(" dt is ", dt, " dx is ", dx)
print(" shape of u ", u.shape)
                                   ##### Begin design matrix construction ########
M  = 20    # M in the paper
B  = 250   # B in the paper
reduced_size = 250 # N in the paper
noise_level  = 0.04  # sigma in the paper

np.random.seed(0)
un = u + noise_level * np.std(u) * np.random.randn(u.shape[0],u.shape[1])

FUn = un.reshape(u.shape[0],101)
uun,usn,uvn = np.linalg.svd(FUn, full_matrices = False)
dim = 15 #u.shape[1] # SVD mode truncation

# reconstructed
un = (uun[:,:dim].dot(np.diag(usn[:dim]).dot(uvn[:dim,:]))).reshape(u.shape[0],101)
DD = 4
PP = 3

u_t, X, description = build_linear_system(un, dt, dx, D=DD, P=PP, time_diff = 'poly', space_diff = 'poly',deg_x = 5, deg_t = 5, width_x = 10, width_t = 10)
description_trunc, X_no_intercept = remove_intercept(X, description) # changed from X to Rn
p = X_no_intercept.shape[1]
subsample_info_iht_d, subsample_info_STR, alphas_sample = stability_selection(X_no_intercept, u_t, reduced_size, M, B, rescale=1.0)

# rescaling lambdas
normalized_alphas_sample = np.zeros((B, M))
for sample in range(0, B):
    normalized_alphas_sample[sample,:] = alphas_sample[sample,:]/alphas_sample[sample,0]
neg_log_stable_lambda = -np.log10(normalized_alphas_sample[0,:])

importance_plot = np.zeros((p, M))
for alpha in range(0, M):
    for i in range(0, p):
        for sample in range(0, B):
            if(subsample_info_iht_d[sample, i, alpha] != 0):
                importance_plot[i,alpha] = importance_plot[i,alpha] + 1
        importance_plot[i,alpha] = importance_plot[i,alpha]/B
        
df = pd.DataFrame(data=importance_plot.astype(np.double))
path = 'IHTd_Burgers_n' + str(noise_level) + '_N'+ str(reduced_size) + '_p' + str(p) + '_dim' + str(dim) + '_B' + str(B) + '.txt'
df.to_csv(path, header=None, index=None, sep=' ', mode='w')
       
importance_plot = np.zeros((p, M))
for alpha in range(0, M):
    for i in range(0, p):
        for sample in range(0, B):
            if(subsample_info_STR[sample, i, alpha] != 0):
                importance_plot[i,alpha] = importance_plot[i,alpha] + 1
        importance_plot[i,alpha] = importance_plot[i,alpha]/B
        
# to comment or uncomment use cmd + /
df = pd.DataFrame(data=importance_plot.astype(np.double))
path = 'STR_Burgers_n' + str(noise_level) + '_N'+ str(reduced_size) + '_p' + str(p) + '_dim' + str(dim) + '_B' + str(B) + '.txt'
df.to_csv(path, header=None, index=None, sep=' ', mode='w')

df = pd.DataFrame(data=neg_log_stable_lambda.astype(np.double))
path = 'lambda_n' + str(noise_level) + '_N'+ str(reduced_size) + '_p' + str(p) + '_dim' + str(dim) + '_B' + str(B) + '.txt'
df.to_csv(path, header=None, index=None, sep=' ', mode='w')
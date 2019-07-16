import numpy as np
from numpy import linalg as LA
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
import itertools
import operator
import math
#%matplotlib notebook
import scipy.io as sio
from operator import itemgetter, attrgetter
import matplotlib.pyplot as plt
from sklearn.linear_model import lars_path, enet_path, lasso_path
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def remove_intercept(X, description):
    
    description_trunc = {}
    for i in range(1, len(description)):
        print("i :" , i, description[i])
        description_trunc[i-1] = description[i]
    
    print(" shape of the truncated description ", len(description_trunc))    
    
    n,d = X.shape
    print("shape",n, d)
    X_no_intercept = np.zeros((n,d-1))
    print("shape of no_intercept", X_no_intercept.shape)
    for i in range(1, d):
        X_no_intercept[:,i-1] = X[:,i] 
    
    return description_trunc, X_no_intercept

def PolyDiff(u, x, deg = 3, diff = 1, width = 5):
    
    """
    u = values of some function
    x = x-coordinates where values are known
    deg = degree of polynomial to use
    diff = maximum order derivative we want
    width = width of window to fit to polynomial

    This throws out the data close to the edges since the polynomial derivative only works
    well when we're looking at the middle of the points fit.
    """

    u = u.flatten()
    x = x.flatten()

    n = len(x)
    du = np.zeros((n - 2*width,diff))

    # Take the derivatives in the center of the domain
    for j in range(width, n-width):

        points = np.arange(j - width, j + width)

        # Fit to a Chebyshev polynomial
        # this is the same as any polynomial since we're on a fixed grid but it's better conditioned :)
        poly = np.polynomial.chebyshev.Chebyshev.fit(x[points],u[points],deg)

        # Take derivatives
        for d in range(1,diff+1):
            du[j-width, d-1] = poly.deriv(m=d)(x[j])

    return du

def build_linear_system(u, dt, dx, D = 4, P = 3,time_diff = 'poly',space_diff = 'poly',lam_t = None,lam_x = None, width_x = None,width_t = None, deg_x = 5,deg_t = None,sigma = 2):
    
    print(" The value of D is changed to 4 ", D)
    
    """ 
    Constructs a large linear system to use in later regression for finding PDE.  
    This function works when we are not subsampling the data or adding in any forcing.

    Input:
        Required:
            u = data to be fit to a pde
            dt = temporal grid spacing
            dx = spatial grid spacing
        Optional:
            D = max derivative to include in rhs (default = 3)
            P = max power of u to include in rhs (default = 3)
            time_diff = method for taking time derivative
                        options = 'poly', 'FD', 'FDconv','TV'
                        'poly' (default) = interpolation with polynomial 
                        'FD' = standard finite differences
                        'FDconv' = finite differences with convolutional smoothing 
                                   before and after along x-axis at each timestep
                        'Tik' = Tikhonov (takes very long time)
            space_diff = same as time_diff with added option, 'Fourier' = differentiation via FFT
            lam_t = penalization for L2 norm of second time derivative
                    only applies if time_diff = 'TV'
                    default = 1.0/(number of timesteps)
            lam_x = penalization for L2 norm of (n+1)st spatial derivative
                    default = 1.0/(number of gridpoints)
            width_x = number of points to use in polynomial interpolation for x derivatives
                      or width of convolutional smoother in x direction if using FDconv
            width_t = number of points to use in polynomial interpolation for t derivatives
            deg_x = degree of polynomial to differentiate x
            deg_t = degree of polynomial to differentiate t
            sigma = standard deviation of gaussian smoother
                    only applies if time_diff = 'FDconv'
                    default = 2
    Output:
        ut = column vector of length u.size
        R = matrix with ((D+1)*(P+1)) of column, each as large as ut
        rhs_description = description of what each column in R is
    """

    n, m = u.shape    #size of u, n is space and m is the time

    if width_x == None: width_x = n/10
    if width_t == None: width_t = m/10
    if deg_t == None: deg_t = deg_x

    # If we're using polynomials to take derviatives, then we toss the data around the edges.
    if time_diff == 'poly': 
        m2 = m-2*width_t
        offset_t = width_t
    else: 
        m2 = m
        offset_t = 0
        
    if space_diff == 'poly': 
        n2 = n-2*width_x
        offset_x = width_x
    else: 
        n2 = n
        offset_x = 0

    if lam_t == None: lam_t = 1.0/m
    if lam_x == None: lam_x = 1.0/n

    ########################
    # First take the time derivaitve for the left hand side of the equation
    ########################
    ut = np.zeros((n2,m2), dtype=np.complex64)

    if time_diff == 'FDconv':
        Usmooth = np.zeros((n,m), dtype=np.complex64)
        # Smooth across x cross-sections
        for j in range(m):
            Usmooth[:,j] = ConvSmoother(u[:,j],width_t,sigma)
        # Now take finite differences
        for i in range(n2):
            ut[i,:] = FiniteDiff(Usmooth[i + offset_x,:],dt,1)

    elif time_diff == 'poly':
        T= np.linspace(0,(m-1)*dt,m)
        for i in range(n2):
            ut[i,:] = PolyDiff(u[i+offset_x,:],T,diff=1,width=width_t,deg=deg_t)[:,0]

    elif time_diff == 'Tik':
        for i in range(n2):
            ut[i,:] = TikhonovDiff(u[i + offset_x,:], dt, lam_t)

    else:
        for i in range(n2):
            ut[i,:] = FiniteDiff(u[i + offset_x,:],dt,1)
    
    ut = np.reshape(ut, (n2*m2,1), order='F')

    ########################
    # Now form the rhs one column at a time, and record what each one is
    ########################

    u2 = u[offset_x:n-offset_x,offset_t:m-offset_t]
    Theta = np.zeros((n2*m2, (D+1)*(P+1)), dtype=np.complex64)
    ux = np.zeros((n2,m2), dtype=np.complex64)
    rhs_description = ['' for i in range((D+1)*(P+1))]

    if space_diff == 'poly': 
        Du = {}
        for i in range(m2):
            Du[i] = PolyDiff(u[:,i+offset_t],np.linspace(0,(n-1)*dx,n),diff=D,width=width_x,deg=deg_x)
    if space_diff == 'Fourier': ik = 1j*np.fft.fftfreq(n)*n
        
    for d in range(D+1):

        if d > 0:
            for i in range(m2):
                if space_diff == 'Tik': ux[:,i] = TikhonovDiff(u[:,i+offset_t], dx, lam_x, d=d)
                elif space_diff == 'FDconv':
                    Usmooth = ConvSmoother(u[:,i+offset_t],width_x,sigma)
                    ux[:,i] = FiniteDiff(Usmooth,dx,d)
                elif space_diff == 'FD': ux[:,i] = FiniteDiff(u[:,i+offset_t],dx,d)
                elif space_diff == 'poly': ux[:,i] = Du[i][:,d-1]
                elif space_diff == 'Fourier': ux[:,i] = np.fft.ifft(ik**d*np.fft.fft(ux[:,i]))
        else: ux = np.ones((n2,m2), dtype=np.complex64) 
            
        for p in range(P+1):
            Theta[:, d*(P+1)+p] = np.reshape(np.multiply(ux, np.power(u2,p)), (n2*m2), order='F')

            if p == 1: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+'u'
            elif p>1: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+'u^' + str(p)
            if d > 0: rhs_description[d*(P+1)+p] = rhs_description[d*(P+1)+p]+\
                                                   'u_{' + ''.join(['x' for _ in range(d)]) + '}'

    return ut, Theta, rhs_description

def build_Theta(data, derivatives, derivatives_description, P, data_description = None):
    """
    builds a matrix with columns representing polynoimials up to degree P of all variables

    This is used when we subsample and take all the derivatives point by point or if there is an 
    extra input (Q in the paper) to put in.

    input:
        data: column 0 is U, and columns 1:end are Q
        derivatives: a bunch of derivatives of U and maybe Q, should start with a column of ones
        derivatives_description: description of what derivatives have been passed in
        P: max power of polynomial function of U to be included in Theta

    returns:
        Theta = Theta(U,Q)
        descr = description of what all the columns in Theta are
    """
    print(" start of build_theta")
    n,d = data.shape
    print("n = ",n, " d = ", d)
    m, d2 = derivatives.shape
    if n != m: raise Exception('dimension error')
    if data_description is not None: 
        if len(data_description) != d: raise Exception('data descrption error')
            
    print("conditions fulfilled")        
    
    # Create a list of all polynomials in d variables up to degree P
    rhs_functions = {}
    f = lambda x, y : np.prod(np.power(list(x), list(y)))
    powers = []            
    for p in range(1,P+1):
            size = d + p - 1
            for indices in itertools.combinations(range(size), d-1):
                starts = [0] + [index+1 for index in indices]
                stops = indices + (size,)
                powers.append(tuple(map(operator.sub, stops, starts)))
    for power in powers: rhs_functions[power] = [lambda x, y = power: f(x,y), power]
    
    print("finished creating poynomials")

    # First column of Theta is just ones.
    Theta = np.ones((n,1), float)
    descr = ['']
    
    # Add the derivaitves onto Theta
    for D in range(1,derivatives.shape[1]):
        Theta = np.hstack([Theta, derivatives[:,D].reshape(n,1)])
        descr.append(derivatives_description[D])
    
    print("finished adding derivatives to the theta_matrix")
        
    # Add on derivatives times polynomials
    for D in range(derivatives.shape[1]):
        for k in rhs_functions.keys():
            func = rhs_functions[k][0]
            #new_column = np.zeros((n,1), dtype=np.complex64)
            new_column = np.zeros((n,1), dtype=float)
            for i in range(n):
                new_column[i] = func(data[i,:])*derivatives[i,D]
            Theta = np.hstack([Theta, new_column])
            if data_description is None: descr.append(str(rhs_functions[k][1]) + derivatives_description[D])
            else:
                function_description = ''
                for j in range(d):
                    if rhs_functions[k][1][j] != 0:
                        if rhs_functions[k][1][j] == 1:
                            function_description = function_description + data_description[j]
                        else:
                            function_description = function_description + data_description[j] + '^' + str(rhs_functions[k][1][j])
                descr.append(function_description + derivatives_description[D])
                            
    return Theta, descr


def Iterative_hard_thresholding_debias(X, y, lasso_lam, max_iter, sub_iter, tol, htp_flag, print_flag=False):
    n,d = X.shape
    coeff_old  = np.zeros((1,d))
    coeff_new  = np.zeros((1,d))
    L = 1.0 
    LL = np.linalg.norm(X.T.dot(X),2)
    one  =  (1./L)*(X.T.dot(y)).reshape(d,)
    XTX  = X.T.dot(X)
  
    coeff_old[0,:] =  one #+ 0.01*np.random.normal(0, 1, d)
    support =  np.arange(0,d).reshape(d,)
    frac_iter = 0
    time_step = (1.0/LL) 
    for iteration in range(0, max_iter):
        temp = np.zeros((1,d))
        if(htp_flag == 1):
            temp[0,:] = coeff_old[0,:] + (2.0/LL) *(one  - XTX.dot(coeff_old[0,:])) #inclusion will make similar algorithm to HTP
        else:
            temp[0,:] = coeff_old[0,:].copy()

        smallinds   = np.where( abs(temp[0,:]) <= lasso_lam/(L))[0]
        biginds     = [i for i in range(d) if i not in smallinds]
        if( len(biginds) == 0 ):  # if( len(biginds)==0 and iteration !=0 ):
            coeff_new[0,:] = 0
            return coeff_new[0,:]
        else:
            #temp    = np.zeros((1,d)) # added this
            temper  = np.zeros((1,len(biginds)))
            grad    = np.zeros((len(biginds),))
            grad_A  = np.zeros((n,))
            LLL     = np.linalg.norm(X[:, biginds].T.dot(X[:, biginds]), 2) 
            XTy_s   = X[:, biginds].T.dot(y).reshape( len(biginds), )
            XTX_s   = X[:, biginds].T.dot( X[:, biginds])
            for k in range(0, sub_iter):
                temp_2      = XTX_s.dot( temp[0, biginds] ).reshape(len(biginds),)
                grad        = (XTy_s - temp_2)
                grad_A      = X[:, biginds].dot(grad)
                if(grad.T.dot(grad) == 0):
                    break
                else:
                    time_step = (grad.T.dot(grad))/(grad_A.T.dot(grad_A))
                    
                temper[0,:] = temp[0,biginds] + (time_step)*grad
                if( np.linalg.norm( y[:,0] - X[:, biginds].dot(temper[0,:])  , 2)**2 < (len(biginds)*lasso_lam) ):
                    coeff_new[0, biginds]   = temper[0,:].copy()
                    coeff_new[0, smallinds] = 0
                    frac_iter = frac_iter + 1
                    break
                
                temp[0, biginds]   = temper[0,:].copy() # copy to old
            
            coeff_new[0,biginds]   = temper[0,:].copy() # new coeffs value
            coeff_new[0,smallinds] = 0                  # new coeffs value
        
        biginds     = np.array(biginds)
        # check for convergence
        if( (np.linalg.norm(coeff_new[0,:] - coeff_old[0,:], np.inf) < tol) and np.all(support == biginds) ):
            if(print_flag):
                print(" converged at iteration ", iteration,  " frac_iter ", frac_iter/(iteration+1), " flag ", htp_flag)
            return coeff_new[0,:]
        
        # copy the values between old and new
        support           = biginds.copy()
        coeff_old[0,:]    = coeff_new[0,:].copy()
        
    return coeff_new[0,:]

def STRidge(X, y, lasso_lam, lam, max_iter):
    
    n,d = X.shape
    clf = linear_model.Ridge(lam, fit_intercept=False, max_iter=5000, tol=1e-8) #+ np.random.normal(0, 1, d)
    clf.fit(X,y)
    coeff = clf.coef_
    num_relevant = d
    for iteration in range(0, max_iter):
        smallinds = np.where( abs(coeff[0,:]) < lasso_lam)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
          
        if(num_relevant == len(new_biginds)):
            break
        else:
            num_relevant = len(new_biginds)
            
        if len(new_biginds) == 0:
            if iteration == 0: 
                #print(" too much tolerance - returned before iteration ", lasso_lam)
                coeff[0,:] = 0
                return coeff
            else: break
                
        biginds = new_biginds
        coeff[0,smallinds] = 0
        if lam != 0: coeff[0,biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0].reshape(len(biginds),)
        else: coeff[0,biginds] = np.linalg.lstsq(X[:, biginds],y)[0].reshape(len(biginds),)
        
    return coeff[0,:]

def stability_selection(X, y, reduced_size, M, B):
    
    p = X.shape[1]
    np.random.seed(60)
    ordered_reduced = np.arange(0, len(y))
    np.random.shuffle(ordered_reduced)
    X_reduced = np.zeros((reduced_size, p))
    y_reduced = np.zeros((reduced_size, 1))
    
    for i in range(0, reduced_size):
        X_reduced[i,:] = X[int(ordered_reduced[i]),:]
        y_reduced[i]   = y[int(ordered_reduced[i])]  # changed 
           
    alphas_sample = np.zeros((B, M))
    subsample_info_lasso = np.zeros((B, p, M))
    subsample_info_STR    = np.zeros((B, p, M))
    subsample_info_iht_d  = np.zeros((B, p, M))
    
    sub_size = int(reduced_size/2)   # the subsize is not complimentary though
    X_sub = np.zeros((B, sub_size, p))
    y_sub = np.zeros((B, sub_size))
    
    ordered_subsample = np.arange(0, reduced_size)  # form an array from 0 to N
    for sample in range(0, B):
        np.random.shuffle(ordered_subsample)
        for j in range(0, sub_size):
            X_sub[sample, j, :] = X_reduced[int(ordered_subsample[j]),:]
            y_sub[sample, j]    = y_reduced[int(ordered_subsample[j]),:]  
    
    Xs_sub = np.zeros((B, sub_size, p))
    ys_sub = np.zeros((B, sub_size)) 
    # standardizing X_sub
    scaler = StandardScaler()
    for sample in range(0, B):
        temp_x = preprocessing.scale(X_sub[sample])
        Xs_sub[sample] = temp_x
        ys_sub[sample] = preprocessing.scale(y_sub[sample],with_std=False)
        
    Xs_sub = Xs_sub.reshape((B, sub_size, p)) 
    ys_sub = ys_sub.reshape((B, sub_size))

    factor      = np.exp(np.log(10)/M)
    weakness    = 0.2
    for sample in range(0, B):
        alphas_lasso, coefs_lasso, _ = lasso_path(Xs_sub[sample], ys_sub[sample], eps=0.1,
                                                      n_alphas=M, fit_intercept=False)
        coefs_lasso = coefs_lasso.reshape(p, M)    
        alphas_sample[sample, :] = alphas_lasso
        for alpha in range(0, M):
            subsample_info_lasso[sample,:,alpha] = coefs_lasso[:,alpha]
 
        for alpha in range(0, M):   
            X = Xs_sub[sample]
            y = ys_sub[sample]
            y = y.reshape(sub_size,1)
            coefs_iht_d = Iterative_hard_thresholding_debias(X, y, alphas_sample[sample,alpha], max_iter=10000, sub_iter=25, tol=1e-8, htp_flag = 1)   # multiplication by 3 is emphirical, 
            subsample_info_iht_d[sample,:,alpha] = coefs_iht_d[:]
            coefs_STR   = STRidge(X, y, 3.0*alphas_sample[sample,alpha], lam = 10**-5, max_iter=10000)
            subsample_info_STR[sample,:,alpha] = coefs_STR[:]
            
        if(sample%5 == 0):
            print(" done with sample ", sample, " size ", reduced_size)
    
    return subsample_info_iht_d, subsample_info_STR, alphas_sample
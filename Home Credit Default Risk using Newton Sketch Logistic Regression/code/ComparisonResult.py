from SketchedNewtonLogistic import *
import timeit
import pandas as pd
import numpy as np

def track_hessian(X_tr,y_tr,path, sketch_type = 'Gaussian', sketch_dim = [], fit_intercept = False, max_iter = 600, 
                  tol = 10**(-5), gmax_iter = 400, random_state = None, penalty = None, lamb_p = 1,a=0.1,b=0.5):
    """
    Compare hessian and sketched hessian at each iteration for different sketch_dim(m)
    
    :param X_tr: training data
    :param y_tr: the response of training data
    :param path: path to store the results
    :param sketch_type: types of randomized sketches {None,'Gaussian','ROS'},default='Gaussian'
    :param sketch_dim: the list of number of rows of the sketch matrix
    :param fit_intercept: whether the intercept term should be included {True,False}, default=False
    :param max_iter: Outer loop maximum iterations, default = 600
    :param tol: Outer loop tolerance, default = 0.00001
    :param gmax_iter: Inner loop maximum iterations, default = 400
    :param random_state: int, RandomState instance, default=None
    :param penalty: {None, 'Ridge', 'Lasso'}, default = None
    :param lamb_p: the parameter of penalty for 'Ridge' and 'Lasso', non-negative number, default=1
    :param a: Scaling factor for line search, default = 0.1
    :param b: Reduction factor for line search, default = 0.5
    """
    distance_nh = []
    for m in sketch_dim:
        print(m, end=' ')
        snm = SketchedNewtonLogistic(sketch_type = sketch_type, sketch_dim = m, fit_intercept = fit_intercept, 
                                     max_iter = max_iter, tol = tol, gmax_iter = gmax_iter, random_state = random_state, 
                                     penalty = penalty, lamb_p = lamb_p, a=a, b=b, hessian_track = True)
        snm.fit(X_tr,y_tr)
        distance_nh += [snm.distance]
        
    pd.DataFrame(distance_nh,index = sketch_dim).T.to_csv(path)
    

def CompareSketchNewton(X_tr, y_tr, X_te, y_te, path, sketch_type = 'Gaussian', sketch_dim=[], fit_intercept = False, max_iter = 600, tol = 10**(-5), gmax_iter = 400, random_state = None, penalty = None, lamb_p = 1,a=0.1,b=0.5):
    """
    Compare the results of different sketch_dim(m)
    
    :param X_tr: training data
    :param y_tr: the response of training data
    :param path: path to store the results
    :param X_te: test data
    :param y_te: the response of test data
    :param sketch_type: types of randomized sketches {None,'Gaussian','ROS'},default='Gaussian'
    :param sketch_dim: the list of number of rows of the sketch matrix
    :param fit_intercept: whether the intercept term should be included {True,False}, default=False
    :param max_iter: Outer loop maximum iterations, default = 600
    :param tol: Outer loop tolerance, default = 0.00001
    :param gmax_iter: Inner loop maximum iterations, default = 400
    :param random_state: int, RandomState instance, default=None
    :param penalty: {None, 'Ridge', 'Lasso'}, default = None
    :param lamb_p: the parameter of penalty for 'Ridge' and 'Lasso', non-negative number, default=1
    :param a: Scaling factor for line search, default = 0.1
    :param b: Reduction factor for line search, default = 0.5
    :param hessian_track: Boolean, whether track the difference between hessian and sketched hessian, default=False
    """
    beta = []
    iteration = []
    runtime = []
    accuracy = []
    
    for m in sketch_dim:
        print(m, end=' ')
        snm = SketchedNewtonLogistic(sketch_type = sketch_type, sketch_dim = m, fit_intercept = fit_intercept, 
                                     max_iter = max_iter, tol = tol, gmax_iter = gmax_iter, random_state = random_state, 
                                     penalty = penalty, lamb_p = lamb_p,a=a,b=b)
        start = timeit.default_timer()
        snm.fit(X_tr,y_tr)
        stop = timeit.default_timer()
        runtime += [stop - start]
        beta += [snm.beta]
        iteration += [snm.iter]
        accuracy += [(snm.predict(X_te) == np.array(y_te)).mean()]
        
    path_beta = path + 'beta.csv'
    path_ira = path + 'ira.csv'
    
    pd.DataFrame(beta,index = sketch_dim).T.to_csv(path_beta)
    pd.DataFrame([iteration,runtime,accuracy],columns = sketch_dim,
             index = ['iteration','runtime','accuracy']).to_csv(path_ira)
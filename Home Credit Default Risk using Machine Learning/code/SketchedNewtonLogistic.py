import numpy as np
import math
from random import choices
from sympy import fwht
import random
import warnings

def GradientLogisticsLossPenalty(X,y,beta0,penalty=None,lamb_p=1):
    """
    Calculates the gradient of the logistic loss or the logistic loss with l2 penalty.
    Their gradients are given by: 
     $$ 
     X^T \left( y - p \right) 
     $$ 
     or 
     $$
     X^T \left( y - p \right) + n*\lambda*\beta^T \beta
     $$ 
     where each entry of $p$ is given by: 
     $$ 
     p_i = \frac{\exp(x_i^T beta)}{1 + \exp(a_i^T beta)} 
     $$ 

    :param X: Design matrix
    :param y: Response variable
    :param beta0: Coefficient
    :param penalty: {None, 'Ridge', 'Lasso'}, default = None
    :param lamb_p: the parameter of penalty for 'Ridge' and 'Lasso', non-negative number, default = 1
    :return: np.ndarray, the negative gradient of the log-likelihood function. 
    """
    eXBeta = np.exp(X @ beta0)
    grad = X.T @ ((eXBeta/(1+eXBeta)) - y)
    if penalty == 'Ridge':
        grad = grad + 2*X.shape[0]*lamb_p*beta0
    return(grad)

def newton_sketch_step(A, x, dx, y, a=0.1, b=0.5,penalty=None,lamb_p=1):
    """
    Backtracking line search
    :param A: Design matrix
    :param x: Coefficients
    :param dx: Step direction
    :param y: Response variable
    :param a: Scaling factor 
    :param b: Reduction factor
    :param penalty: {None, 'Ridge', 'Lasso'}, default = None
    :param lamb_p: the parameter of penalty for 'Ridge' and 'Lasso', non-negative number, default = 1
    :return: float, updated beta
    """
    mu = 1
    G = GradientLogisticsLossPenalty(X=A,y=y,beta0=x,penalty=penalty,lamb_p = lamb_p).reshape(1,-1)
    AX = A@x
    L1 = np.sum(np.log(1 + np.exp(AX)) - y * AX)
    C = mu * a * G@dx
    nx = x + mu *dx
    AnX = A@nx
    L2 = np.sum(np.log(1 + np.exp(AnX)) - y * AnX)
    while (L1 + C< L2):
        mu = mu * b
        nx = x + mu *dx
        AnX = A@nx
        L2 = np.sum(np.log(1 + np.exp(AnX)) - y * AnX)
        C = mu * a * G@dx
    return nx

def SketchedMatrix(X,m,sketch_type='Gaussian',random_state=None):
    """
    Generate a sketch matrix S and calculate S@X
    :param X: Design matrix
    :param m: Number of rows of the sketch matrix, non-negative integer
    :param sketch_type: {'Gaussian','ROS'}, default='Gaussian'. If the sample size is not power of two, we could not use 'ROS'.
    :param random_state: int, RandomState instance, default=None
    :return: ndarray, S@X
    """
    n,p = X.shape
    if random_state is None:
        if sketch_type == 'Gaussian':
            SM = np.random.randn(m,n)
            FM = SM @ X
            return(FM)
        elif sketch_type == 'ROS':
            # check if sample size is power of two
            if math.floor(np.log2(n)) == np.log2(n):
                S = choices(list(np.arange(0,n)),k=m)
                D = choices(list([-1,1]),k=n)
                HDX = np.zeros((n,p))
                for i in range(p):
                    HDX[:,i] = fwht(D*X[:,i])
                SHDX = np.zeros((m,p))
                for i in range(m):
                    SHDX[i,] =  HDX[S[i],]
                return(SHDX)
            else:
                print('The dimension is wrong.')
        else:
            print('The sketch_type is wrong.')
    else:
        if sketch_type == 'Gaussian':
            np.random.seed(random_state)
            SM = np.random.randn(m,n)/(m**0.5)
            FM = SM @ X
            return(FM)
        elif sketch_type == 'ROS':
            # check if sample size is power of two
            if math.floor(np.log2(n)) == np.log2(n):
                random.seed(random_state)
                S = choices(list(np.arange(0,n)),k=m)
                random.seed(random_state)
                D = choices(list([-1,1]),k=n)
                HDX = np.zeros((n,p))
                for i in range(p):
                    HDX[:,i] = fwht(D*X[:,i])
                SHDX = np.zeros((m,p))
                for i in range(m):
                    SHDX[i,] =  HDX[S[i],]/(m**0.5)
                return(SHDX)
            else:
                print('The dimension is wrong.')
        else:
            print('The sketch_type is wrong.')


def SketchedHessianHalf(X,beta0,m,sketch_type='Gaussian',penalty = None, lamb_p = 1, random_state=None):
    """
    Calculate H^{1/2} and S@H^{1/2}, where S is the sketched matrix and H is the Hessian matrix.
    :param X: Design matrix
    :param beta0: Coefficient
    :param m: Number of rows of the sketch matrix, non-negative integer
    :param sketch_type: {'Gaussian','ROS'}, default='Gaussian'
    :param penalty: {None, 'Ridge', 'Lasso'},default = None
    :param lamb_p: the parameter of penalty for 'Ridge' and 'Lasso', non-negative number, default=1
    :param random_state: int, RandomState instance, default=None
    :return: tuple, H^{1/2} and S@H^{1/2}
    """
    if penalty == 'Ridge':
        # When considering the l2 penalized logistic regression model, we implement fully sketched Newton update instead of partially sketched Newton update.
        eXBeta = np.exp(X @ beta0)
        Wh = np.concatenate([(eXBeta/((1+eXBeta)**2))**0.5,
                            np.ones(len(beta0))*np.sqrt(2*lamb_p*X.shape[0])],axis = 0).reshape(-1,1)
        nX = np.concatenate([X,np.identity(len(beta0))],axis=0)
        Heh = Wh*nX # Hessian matrix square root
    elif penalty is None:
        eXBeta = np.exp(X @ beta0)
        Wh = ((eXBeta/((1+eXBeta)**2))**0.5).reshape(-1,1)
        Heh = Wh*X
    if sketch_type is None: 
        sHeh = Heh
    else:
        # Sketched Hessian matrix square root
        sHeh = SketchedMatrix(Heh,m,sketch_type=sketch_type,random_state=random_state)  
    return(Heh,sHeh)



def _cg(fhess_p, fgrad, maxiter, tol):
    """
    Solve iteratively the linear system 'fhess_p . xsupi = fgrad' with a conjugate gradient descent. Adapted from sklearn.
    :param fhess_p : ndarray hessian
    :param fgrad : ndarray, shape (n_features,) or (n_features + 1,) Gradient vector
    :param maxiter : int, Number of CG iterations.
    :param tol : float, Stopping criterion.
    :return: ndarray, shape (n_features,) or (n_features + 1,), Estimated solution
    """
    xsupi = np.zeros(len(fgrad), dtype=fgrad.dtype)
    ri = fgrad
    psupi = -ri
    i = 0
    dri0 = np.dot(ri, ri)

    while i <= maxiter:
        if np.sum(np.abs(ri)) <= tol:
            break

        Ap = fhess_p@psupi
        # check curvature
        curv = np.dot(psupi, Ap)
        if 0 <= curv <= 3 * np.finfo(np.float64).eps:
            break
        elif curv < 0:
            if i > 0:
                break
            else:
                # fall back to steepest descent direction
                xsupi += dri0 / curv * psupi
                break
        alphai = dri0 / curv
        xsupi += alphai * psupi
        ri = ri + alphai * Ap
        dri1 = np.dot(ri, ri)
        betai = dri1 / dri0
        psupi = -ri + betai * psupi
        i = i + 1
        dri0 = dri1          # update np.dot(ri,ri) for next time.
    return xsupi


class SketchedNewtonLogistic:
    """
    Sketched Newton Method for Logistic Regression and L2-penalized Logistic Regression
    
    :param sketch_type: types of randomized sketches {None,'Gaussian','ROS'},default='Gaussian'
    :param sketch_dim: number of rows of the sketch matrix {N+}
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
    
    def __init__(self,sketch_type = 'Gaussian', sketch_dim = None, fit_intercept = False, max_iter = 600, tol = 10**(-5), gmax_iter = 400, random_state = None, penalty = None, lamb_p = 1,a=0.1,b=0.5, hessian_track = False):

        self.sketch_type = sketch_type
        self.m = sketch_dim
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter #outer loop
        self.tol = tol #tolerance 
        self.gmax_iter = gmax_iter #inner loop
        self.random_state = random_state
        self.penalty = penalty
        self.lamb_p = lamb_p
        self.a = a
        self.b = b
        self.hessian_track = hessian_track
        self.iter = -1 #number of total outer iteration
        
    def fit(self, X, y, beta0 = None):
        """
        Fit the model according to the given training data.
        :param X: Design matrix
        :param y: response variable
        :param beta0: Coefficient 
        :return: self Fitted estimator
        """
        
        # check whether to fit the intercept
        if self.fit_intercept == True:
            X = np.concatenate([np.ones(X.shape[0]).reshape(-1,1),X],axis = 1)
            
        self.n,self.p = X.shape
        # if number of rows of the sketch matrix is not given, using the following criteria
        if self.m is None:
            self.m = max(math.floor((self.n)*3/5),self.p)
        
        # set initial beta 
        if beta0 is None:
            beta0 = np.zeros(self.p)
        elif len(beta0) == self.p:
            beta0 = beta0
        else:
            print('The dimension of beta0 is wrong.')    
        self.beta = beta0
        xk = beta0
        
        # store distance between sketched hessian and true hessian
        self.distance = []
        
        for i in range(self.max_iter):
            # calculate sketched hessian
            if self.random_state is None:
                Heh,sHeh = SketchedHessianHalf(X,beta0,self.m,sketch_type=self.sketch_type,penalty = self.penalty,lamb_p = self.lamb_p,random_state= None)
            else:
                Heh,sHeh = SketchedHessianHalf(X,beta0,self.m,sketch_type=self.sketch_type,penalty = self.penalty,lamb_p = self.lamb_p,random_state=self.random_state+i*5)
            W = sHeh.T@sHeh
            
            # calculate true hessian
            if self.hessian_track == True:
                He = Heh.T @ Heh
                #calculate distance between sketched hessian and true hessian
                s = np.sum((He-W)**2)
                self.distance += [s]
                
            # Minimization of scalar function of one or more variables using the Newton conjugate gradient descent algorithm.
            fgrad = GradientLogisticsLossPenalty(X,y,beta0,penalty = self.penalty,lamb_p =self.lamb_p)
            fhess_p = W
          
            if any([(self.penalty is None),(self.penalty == 'Ridge')]):
                absgrad = np.abs(fgrad)
                if np.max(absgrad) <= self.tol:
                    break

                maggrad = np.sum(absgrad)
                eta = min([0.5, np.sqrt(maggrad)])
                termcond = eta * maggrad

                # Inner loop: solve the Newton update by conjugate gradient, to
                # avoid inverting the Hessian
                xsupi = _cg(fhess_p, fgrad, maxiter=self.gmax_iter, tol=termcond)
                # Backtracking line search
                xk = newton_sketch_step(X,xk,xsupi,y,self.a,self.b,penalty=self.penalty,lamb_p=self.lamb_p)      

                if i == self.max_iter-1:
                    self.iter = self.max_iter
                    print("Failed to converge")

                beta0=xk
            elif self.penalty  == 'Lasso':
                nbeta = beta0
                nbeta_o = nbeta
                for j in range(self.gmax_iter):
                    for k in range(p):
                        s = list(set(np.arange(0,k)).union(set(np.arange(k+1,self.p))))
                        z = -grad[k]-2*W[k,s]@((nbeta-beta0)[s])+2*W[k,k]*beta0[k]
                        nbeta[k] =  SoftThresholding(z,self.lamb_p*n)/(2*W[k,k])
                beta0 = nbeta
            else:
                print('The parameter, penalty, is wrong.')      
            if np.linalg.norm(beta0-self.beta) < self.tol:
                self.beta = beta0
                self.iter =i+1
                break
            else:
                self.beta = beta0 # update beta
                
    def predict(self,X_test):
        """
        Prodict the model according to the fitted estimator
        :param X_test: Test data set 
        :return: self predict result
        """
        if self.fit_intercept == True:
            X_test = np.concatenate([np.ones(X_test.shape[0]).reshape(-1,1),X_test],axis = 1)
            
        e = np.exp(X_test @ self.beta)
        self.probs = e/(1+e) # probability
        self.preclass = self.probs>0.5 # predict result
        return(self.preclass)
        
        
        
        
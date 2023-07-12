# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:56:02 2023

@author: Cristobal
"""

import numpy as np
import proxop as pr
def P_cdom_star(x):
    """ 
    Calculate the projection onto the closure of the domain of the Fenchel
    conjugate of the exponential function.
    
    Parameters
    ----------
    x   : float
        Real numbers.
        
    Returns
    -------
    output : float
        Projection onto the closure of the domain
    """
    if x >= 0:
        return x
    else:
        return float(0)
    
def P_cdom_persp(x,eta):
    """ 
    Calculate the projection onto the closure of the domain of the perspective
    of the exponential function.
    
    Parameters
    ----------
    x   : float
    eta : float
        Real numbers.
        
    Returns
    -------
    output : list
        list that contains the projection onto the closure of the domain of
        the perspective of the exponential function.
    """
    if eta >= 0:
        return [x,eta]
    else:
        return [x,0]
def f_star(x):
    """ 
    Calculate the evaluation of the perspective function of the exponential
    function.
    
    Parameters
    ----------
    x   : float
    eta : float
        Real numbers.
        
    Returns
    -------
    output : float
        Evaluation of the perspective of the exponential function.
    """
    if x>0:
        return x*np.log(x) - x 
    elif x == 0:
        return float(0)
    else:
        return np.inf
    
def f_persp(x,eta):
    """ 
    Calculate the evaluation of the perspective function of the exponential
    function.
    
    Parameters
    ----------
    x   : float
    eta : float
        Real numbers.
        
    Returns
    -------
    output : float
        Evaluation of the perspective of the exponential function.
    """
    if eta > 0:
        return eta*np.exp(x/eta)
    elif eta == 0 and x <=0:
        return 0 
    else:
        return np.inf
    
def prox_f_star(x, Gamma):
    """ 
    Calculate the evaluation of the proximity operator for the 
    Fenchel conjugate of the exponential function.
    
    Parameters
    ----------
    x     : float
        Real number
    Gamma : float 
        Real number greater than 0.
        
    Returns
    -------
    output : float
        Evaluation of the proximity operator.
    """
    if Gamma < 0.0:
        raise ValueError(
            "'gamma' in prox_{gamma * e^*} "
            + "must be greater or equal than 0"
        )
    if Gamma == 0.0:
        return P_cdom_star(x)
    else:
        return (x- Gamma*pr.Exp().prox([x/Gamma], gamma=float(1/Gamma)))[0]

def bounds(phi,M,sigma,zero_tol):
    aux = phi(M)
    if abs(aux) < zero_tol:
        return [M,-1 -1]
    if aux <= 0:
        while aux <= 0:
            if abs(aux) < zero_tol:
                return [M,-1,-1]
            M *= sigma
            aux = phi(M)
        return [M, M/sigma,1]
    if aux >= 0:
        while aux >= 0:
            if abs(aux) < zero_tol:
                return [M,-1,-1]
            M *= (1/sigma)
            aux = phi(M)
        return [M*sigma, M,1]
    
# def bisection(phi,mu_d,mu_u,eps):
#     mu_star = (mu_d + mu_u)/2
#     aux = phi(mu_star)
#     while abs(aux) > eps:
#         if aux > 0:
#             mu_u = mu_star
#         else:
#             mu_d = mu_star
#         mu_star = (mu_d + mu_u)/2
#         aux = phi(mu_star)
#     return mu_star
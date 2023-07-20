# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 11:54:55 2023

@author: Cristobal
"""
from prelim import P_cdom_star,P_cdom_persp,f_star,prox_f_star,bounds
from scipy.optimize import root_scalar
#%%
def prox_f_persp(x,eta,Gamma,M,sigma,
                 zero_tol = 10**(-16),
                 alg = "brentq"):
    """ 
    Calculate the evaluation of the proximity operator for the 
    perspective of the exponential function.
    
    Parameters
    ----------
    x       : float
        Real number.
    eta     : float
        Real number.
    Gamma   : float 
        Real number greater than 0.
    mu_d    : float
        Real number greater than 0. This represents a lower initial candidate
        for the bisection algorithm in this function
    mu_u    : float
        Real number greater than 0. This represents an upper initial candidate
        for the bisection algorithm in this function
    zero_tol: float
        Real number greater than 0, anything less than this number is
        considered equal to 0 in the function
    Returns
    -------
    output  : float
        Evaluation of the proximity operator.
    """
    
    if Gamma < 0.0:
        raise ValueError(
            "'gamma' in prox_{gamma * e^*} "
            + "must be greater or equal than 0"
        )
    if Gamma < zero_tol:
        return P_cdom_persp(x, eta)
    P = P_cdom_star(x/Gamma)
    l = eta + Gamma * f_star(P)
    prox_star = lambda mu : prox_f_star(x/Gamma, mu/Gamma, zero_tol)
    phi = lambda mu : mu - eta - Gamma*f_star(prox_f_star(x/Gamma,
                                                          mu/Gamma,
                                                          zero_tol
                                                          ))
    if l < zero_tol:
        return [x - Gamma*P, 0]
    else:
        if x >= 0 :
            mu_u = l
            mu_d = 0
            if abs(phi(mu_u)) < zero_tol:
                return [x - Gamma*prox_star(mu_u), mu_u]
            if abs(phi(mu_d)) < zero_tol:
                return [x - Gamma*prox_star(mu_d), mu_d]
        else: 
            bound = bounds(phi, M, sigma, zero_tol)
            if bound[2] == -1:
                return [x - Gamma*prox_star(bound[0]), bound[0]] 
            else:
                mu_d = bound[0]
                mu_u = bound[1]
        if alg == "newton":
            mu_star = root_scalar(phi,
                                  x0 = M,
                                  method = alg).root
        if alg == "secant":
            mu_star = root_scalar(phi,
                                  x0 = M,
                                  x1 = M/sigma,
                                  method = alg).root
        else:
            mu_star = root_scalar(phi,
                                  bracket = [mu_d,mu_u],
                                  method = alg).root
        return [x - Gamma*prox_star(mu_star), mu_star]
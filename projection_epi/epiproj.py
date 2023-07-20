# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:26:16 2023

@author: Cristobal
"""

import math as mt
#import numpy as np
from prelim import P_cdom_persp,f_persp,bounds
from f_persp_prox import f_persp_prox 
from prox_perspective import prox_f_persp
from scipy.optimize import root_scalar
#%%
def proj_epi(x,eta,delta,M,sigma,
             zero_tol = 10**(-15),
             alg = "brentq"
             ):
    P = P_cdom_persp(x, eta)
    l = -delta + f_persp(P[0],P[1])
    prox_mu = lambda mu : prox_f_persp(x,eta, mu,M,sigma)
    phi = lambda mu : mu + delta - f_persp_prox(x,eta,mu,M,sigma)
    if l < zero_tol:
        return [P[0],P[1],delta]
    else:
        if (eta > 0 or (eta <= 0 and x <= 0)) and l != mt.inf:
            mu_u = l
            mu_d = 0
            if abs(phi(mu_u)) < zero_tol:
                mu_star = mu_u
                aux_prox = prox_mu(mu_star)
                return [aux_prox[0],aux_prox[1],delta + mu_star]
            if abs(phi(mu_d)) < zero_tol:
                mu_star = mu_d
                aux_prox = prox_mu(mu_star)
                return [aux_prox[0],aux_prox[1],delta + mu_star]
            
        else: 
            bound = bounds(phi, M, sigma, zero_tol)
            if bound[2] == -1:
                mu_star = bound[0]
                aux_prox = prox_mu(mu_star)
                return [aux_prox[0],aux_prox[1],delta + mu_star]
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
            #print("[mu_d,mu_u]=",[mu_d,mu_u])
            mu_star = root_scalar(phi,
                                  bracket = [mu_d,mu_u],
                                  method = alg).root
        aux_prox = prox_mu(mu_star)
        return [aux_prox[0],aux_prox[1],delta + mu_star]

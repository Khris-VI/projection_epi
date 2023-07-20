# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:01:48 2023

@author: Cristobal
"""
import proxop as pr
from prelim import P_cdom_star,P_cdom_persp,f_star,f_persp,prox_f_star,bounds
from scipy.optimize import root_scalar
#%%
def f_persp_prox(x,eta,Gamma,M,sigma,
                 zero_tol = 10**(-16),
                 alg = "brentq"):
    
    if Gamma < 0.0:
        raise ValueError(
            "'gamma' in prox_{gamma * e^*} "
            + "must be greater or equal than 0"
        )
    if Gamma < zero_tol:
        P = P_cdom_persp(x, eta)
        return f_persp(P[0],P[1])
    P = P_cdom_star(x/Gamma)
    l = eta + Gamma * f_star(P)
    phi = lambda nu : nu - eta - Gamma*f_star(prox_f_star(x/Gamma,
                                                          nu/Gamma,
                                                          zero_tol))
    
    #Sign check for the eta + Gamma *f_star(P_cdom_star(x/Gamma))
    if l < zero_tol:
        return 0.0
    else:
        #Projection onto closure is in the domain
        if x >= 0 :
            nu_u = l
            nu_d = 0
            if abs(phi(nu_u)) < zero_tol:
                nu_star = nu_u
                return nu_star*pr.Exp()(pr.Exp().prox(x/nu_star,
                                                      gamma=Gamma/nu_star))
            if abs(phi(nu_d)) < zero_tol:
                nu_star = nu_u
                return nu_star*pr.Exp()(pr.Exp().prox(x/nu_star,
                                                      gamma=Gamma/nu_star))
        #Projection onto closure is not in the domain 
        else: 
            bound = bounds(phi, M, sigma, zero_tol)
            if bound[2] == -1:
                nu_star = bound[0]
                return nu_star*pr.Exp()(pr.Exp().prox(x/nu_star,
                                                      gamma=Gamma/nu_star))
            else:
                nu_d = bound[0]
                nu_u = bound[1]
        #Root Finding algorithm check
        if alg == "newton":
            nu_star = root_scalar(phi,
                                  x0 = M,
                                  method = alg).root
        if alg == "secant":
            nu_star = root_scalar(phi,
                                  x0 = M,
                                  x1 = M/sigma,
                                  method = alg).root
        else:
            nu_star = root_scalar(phi,
                                  bracket = [nu_d,nu_u],
                                  method = alg).root
        return nu_star*pr.Exp()(pr.Exp().prox(x/nu_star, gamma=Gamma/nu_star))

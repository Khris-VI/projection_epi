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
                 nu_d = float(10**(-10)),
                 nu_u = float(10**5),
                 zero_tol = 10**(-16),
                 alg = "brentq"):
    if Gamma == 0:
        P = P_cdom_persp(x, eta)
        return f_persp(P[0],P[1])
    P = P_cdom_star(x/Gamma)
    l = eta + Gamma * f_star(P)
    phi = lambda nu : nu - eta - Gamma*f_star(prox_f_star(x/Gamma,nu/Gamma))
    if l <= 0:
        return 0.0
    else:
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
        else: 
            bound = bounds(phi, M, sigma, zero_tol)
            if bound[2] == -1:
                nu_star = bound[0]
                return nu_star*pr.Exp()(pr.Exp().prox(x/nu_star,
                                                      gamma=Gamma/nu_star))
            else:
                nu_d = bound[1]
                nu_u = bound[0]
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

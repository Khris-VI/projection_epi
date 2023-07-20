# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 18:23:52 2023

@author: Cristobal
"""
#import math as mt
import numpy as np
from prelim import P_cdom_persp,f_persp,bounds
from f_persp_prox import f_persp_prox 
from prox_perspective import prox_f_persp
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar
#from proxop.utils.lambert_W import lambert_W
import time
import random
#%%

#%%
# =============================================================================
# def proj_epi(x,
#              eta,
#              delta,
#              M,
#              sigma,
#              mu_d = float(10**(-10)),
#              mu_u = float(10**10),
#              max_iter = 100,
#              zero_tol = 10**(-15),
#              alg = "brentq"
#              ):
#     P = P_cdom_persp(x, eta)
#     l = -delta + f_persp(P[0],P[1])
#     prox_mu = lambda mu : prox_f_persp(x,eta, mu)
#     phi = lambda mu : mu + delta - f_persp_prox(x,eta,mu)
#     if l <= 0:
#         return [P[0],P[1],delta]
#     else:
#         if eta > 0 or (eta <= 0 and x <= 0) :
#             mu_u = l
#             mu_d = 0
#         else: 
#             aux = phi(mu_u)
#             while aux <= 0.0:
#                 if abs(aux) <= zero_tol:
#                     aux_prox = prox_mu(mu_u)
#                     return [aux_prox[0],aux_prox[1],delta + mu_u]
#                 mu_u *= 10
#                 aux = phi(mu_u)
#             aux = phi(mu_d)
#             print("aux = ", aux)
#             while aux >= 0.0:
#                 if abs(aux) <= zero_tol:
#                     aux_prox = prox_mu(mu_d)
#                     return [aux_prox[0],aux_prox[1],delta + mu_d]
#                 mu_d *= 10**(-1)
#                 aux = phi(mu_d)
#         if abs(phi(mu_u)) <= zero_tol:
#             aux_prox = prox_mu(mu_u)
#             return [aux_prox[0],aux_prox[1],delta + mu_u]
#         if abs(phi(mu_d)) <= zero_tol:
#             aux_prox = prox_mu(mu_d)
#             return [aux_prox[0],aux_prox[1],delta + mu_d]
#         print("phi(mu_d) = ",phi(mu_d))
#         print("phi(mu_u) = ",phi(mu_u))
#         mu_star = bisect(phi,mu_d,mu_u,maxiter = max_iter)
#         aux_prox = prox_mu(mu_star)
#         return [aux_prox[0],aux_prox[1],delta + mu_star]
# =============================================================================
#%%
def proj_epi(x,eta,delta,M,sigma,
             mu_d = float(10**(-10)),
             mu_u = float(10**10),
             max_iter = 100,
             zero_tol = 10**(-15),
             alg = "brentq"
             ):
    P = P_cdom_persp(x, eta)
    l = -delta + f_persp(P[0],P[1])
    prox_mu = lambda mu : prox_f_persp(x,eta, mu,M,sigma)
    phi = lambda mu : mu + delta - f_persp_prox(x,eta,mu,M,sigma)
    if l <= 0:
        return [P[0],P[1],delta]
    else:
        if eta > 0 or (eta <= 0 and x <= 0) :
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
                mu_d = bound[1]
                mu_u = bound[0]
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
        aux_prox = prox_mu(mu_star)
        return [aux_prox[0],aux_prox[1],delta + mu_star]
#%%
N = 1000
n = 20
x = [random.randint(-n, n)*random.random() for i in range(0,N)]
eta = [random.randint(-n, n)*random.random() for i in range(0,N)]
delta = [random.randint(-n, n)*random.random() for i in range(0,N)] 
times = []
t_0 = time.time()
for i in range(0,N):
    t_p = time.time()
    print ("\nx=",x[i],
            "\neta = ",eta[i],
            "\ndelta = ", delta[i],
            "\ni = ",i,
            )
    print("\nproj_epi(x,eta,delta) = ",proj_epi(x[i],
                                          eta[i],
                                          delta[i],
                                          1,
                                          2,
                                          )
          )
    t_pf = time.time()
    times.append(t_pf-t_p)
t_f = time.time()
print("\nelapsed time =",t_f - t_0,"seconds")
print("\naverage elapsed time =","{:.2e}".format(sum(times)/len(times)),
      "seconds")
#%%
n = 7207
print ("\nx=",x[n],
        "\neta = ",eta[n],
        "\ndelta = ", delta[n],
        "\ni = ",n,
      )
t_0 = time.time()
print("\nproj_epi(x,eta,delta) = ",proj_epi(x[n],
                                            eta[n],
                                            delta[n],
                                            1,
                                            2,
                                            max_iter = 700)
        )
t_f = time.time()
print("elapsed time =",t_f - t_0,"seconds")
#%%
X = np.linspace(0, 10**(-10), 1000)
f = lambda mu: mu + delta[n] - f_persp_prox(x[n],eta[n],mu,1,3)
Y = np.array([f(X[i]) for i in range(0,X.size)])
delta[n]
#%%
plt.plot(X, Y, color='red')

plt.show()

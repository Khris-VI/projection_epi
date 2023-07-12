# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 18:23:52 2023

@author: Cristobal
"""
#import numpy as np
import math as mt
import numpy as np
import proxop as pr
from prelim import P_cdom_star,P_cdom_persp,f_star,f_persp,prox_f_star,bounds
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar
#from proxop.utils.lambert_W import lambert_W
import time
import random
#%%
def prox_f_persp(x,
                 eta,
                 Gamma,
                 M,
                 sigma,
                 mu_d = float(10**(-10)),
                 mu_u = float(10**5),
                 zero_tol = 10**(-16),
                 alg = "brentq"
                 ):
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
    if Gamma == 0:
        return P_cdom_persp(x, eta)
    P = P_cdom_star(x/Gamma)
    l = eta + Gamma * f_star(P)
    prox_star = lambda mu : prox_f_star(x/Gamma, mu/Gamma)
    phi = lambda mu : mu - eta - Gamma*f_star(prox_f_star(x/Gamma, mu/Gamma))
    if l <= 0:
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
        return [x - Gamma*prox_star(mu_star), mu_star]
#%%
N = 1000
x_array = [random.randint(-20, 20)*random.random() for i in range(0,N)]
eta_array = [random.randint(-20, 20)*random.random() for i in range(0,N)]
Gamma_array = [random.randint(1, 20)*random.random() for i in range(0,N)]
times = []
t_0 = time.time()
for i in range(0,N):
    t_p = time.time()
    print ("\nx=",x_array[i],
            "\neta = ",eta_array[i],
            "\nGamma = ", Gamma_array[i],
            "\ni = ",i,
            "\nprox_f_persp(x,eta,Gamma) = ",prox_f_persp(x_array[i],
                                                          eta_array[i],
                                                          Gamma_array[i],
                                                          0.1,
                                                          2,
                                                          alg = "brentq")
            )
    t_pf = time.time()
    times.append(t_pf-t_p)
t_f = time.time()
print("\nelapsed time =",t_f - t_0,"seconds")
print("\naverage elapsed time =","{:.2e}".format(sum(times)/len(times)),
      "seconds")
#%%
def f_persp_prox(x,
                 eta,
                 Gamma,
                 M,
                 sigma,
                 nu_d = float(10**(-10)),
                 nu_u = float(10**5),
                 zero_tol = 10**(-16),
                 alg = "brentq"
                 ):
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
#%%
N = 1000
x_array = [random.randint(-100, 100)*random.random() for i in range(0,N)]
eta_array = [random.randint(-100, 100)*random.random() for i in range(0,N)]
Gamma_array = [random.randint(1, 100)*random.random() for i in range(0,N)]
times = []
t_0 = time.time()
for i in range(0,N):
    t_p = time.time()
    print ("\nx=",x_array[i],
            "\neta = ",eta_array[i],
            "\nGamma = ", Gamma_array[i],
            "\ni = ",i,
            "\nprox_f_persp(x,eta,Gamma) = ",f_persp_prox(x_array[i],
                                                          eta_array[i],
                                                          Gamma_array[i],
                                                          0.1,
                                                          2,
                                                          alg = "brentq")
            )
    t_pf = time.time()
    times.append(t_pf-t_p)
t_f = time.time()
print("\nelapsed time =",t_f - t_0,"seconds")
print("\naverage elapsed time =","{:.2e}".format(sum(times)/len(times)),
      "seconds")
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
x = [random.randint(-20, 20)*random.random() for i in range(0,N)]
eta = [random.randint(-20, 20)*random.random() for i in range(0,N)]
delta = [random.randint(-20, 20)*random.random() for i in range(0,N)] 
times = []
t_0 = time.time()
for i in range(0,N):
    tp = time.time()
    print ("\nx=",x[i],
            "\neta = ",eta[i],
            "\ndelta = ", delta[i],
            "\ni = ",i,
            )
    print("\nproj_epi(x,eta,delta) = ",proj_epi(x[i],
                                          eta[i],
                                          delta[i],
                                          .5,
                                          3,
                                          max_iter = 700)
          )
    t_pf = time.time()
    times.append(t_pf-t_p)
t_f = time.time()
print("\nelapsed time =",t_f - t_0,"seconds")
print("\naverage elapsed time =","{:.2e}".format(sum(times)/len(times)),
      "seconds")
#%%
n = 603
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
                                            3,
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

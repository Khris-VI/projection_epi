# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 18:23:52 2023

@author: Cristobal
"""
#import numpy as np
import math as mt
import proxop as pr
#from matplotlib import pyplot as plt
from scipy.optimize import bisect
from scipy.optimize import newton
#from proxop.utils.lambert_W import lambert_W
import time
import random
#%%
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
        return x*mt.log(x) - x 
    elif x == 0:
        return float(0)
    else:
        return mt.inf
    
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
        return eta*mt.exp(x/eta)
    elif eta == 0 and x <=0:
        return 0 
    else:
        return mt.inf
    
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
#%%
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
#%%
def prox_f_persp(x,
                 eta,
                 Gamma,
                 M,
                 sigma,
                 mu_d = float(10**(-10)),
                 mu_u = float(10**5),
                 zero_tol = 10**(-16)
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
        else: 
            bound = bounds(phi, M, sigma, zero_tol)
            if bound[2] == -1:
                return [x - Gamma*prox_star(bound[0]), bound[0]] 
            else:
                mu_d = bound[1]
                mu_u = bound[0]
        mu_star = bisect(phi,mu_d,mu_u)
        return [x - Gamma*prox_star(mu_star), mu_star]

#%%
N = 1000
x_array = [random.randint(-20, 20)*random.random() for i in range(0,N)]
eta_array = [random.randint(-20, 20)*random.random() for i in range(0,N)]
Gamma_array = [random.randint(1, 20)*random.random() for i in range(0,N)]
#%% 
t_0 = time.time()
for i in range(0,N):
    print ("\nx=",x_array[i],
            "\neta = ",eta_array[i],
            "\nGamma = ", Gamma_array[i],
            "\ni = ",i,
            "\nprox_f_persp(x,eta,Gamma) = ",prox_f_persp(x_array[i],
                                                          eta_array[i],
                                                          Gamma_array[i],
                                                          1,
                                                          2)
            )
t_f = time.time()
print("\nelapsed time =",t_f - t_0,"seconds")
#%%
t_0 = time.time()
for i in range(0,N):
    prox_f_persp(x_array[i],eta_array[i],Gamma_array[i],1,2)
t_f = time.time()
print("\nelapsed time =",t_f - t_0,"seconds")
#%%
x = x_array[11]
eta = eta_array[11]
Gamma = Gamma_array[1]
t_0 = time.time()
print(prox_f_persp(x_array[i],eta_array[i],Gamma_array[i],1,2))
t_f = time.time()
print("\nelapsed time =",t_f - t_0,"seconds")
#%%
def f_persp_prox(x,
                 eta,
                 Gamma,
                 nu_d = float(10**(-10)),
                 nu_u = float(10**5),
                 zero_tol = 10**(-16)):
    P = P_cdom_star(x/Gamma)
    l = eta + Gamma * f_star(P)
    prox_star = lambda nu : prox_f_star(x/Gamma, nu/Gamma)
    phi = lambda nu : nu - eta - Gamma*f_star(prox_f_star(x/Gamma, nu/Gamma))
    if l <= 0:
        return 0.0
    else:
        if x >= 0 :
            nu_u = l
        else: 
            aux_prox = prox_star(nu_u)
            aux = nu_u - eta - Gamma*f_star(aux_prox)
            while aux <=0:
                if abs(aux) <= zero_tol:
                    return nu_u*pr.Exp()(pr.Exp().prox(x/nu_u,
                                                          gamma=Gamma/nu_u))
                nu_u *= 10
                aux_prox = prox_star(nu_u)
                aux = nu_u - eta - Gamma*f_star(aux_prox)
            aux_prox = prox_star(nu_d)
            aux = nu_d - eta - Gamma*f_star(aux_prox)
            while aux >= 0:
                if abs(aux) <= zero_tol:
                    return nu_d*pr.Exp()(pr.Exp().prox(x/nu_d,
                                                          gamma=Gamma/nu_d))
                nu_d *= 10**(-1)
                aux_prox = prox_star(nu_d)
                aux = nu_d - eta - Gamma*f_star(aux_prox) 
        nu_star = bisect(phi,nu_d,nu_u)
        return nu_star*pr.Exp()(pr.Exp().prox(x/nu_star, gamma=Gamma/nu_star))

#%%
def proj_epi(x,
             eta,
             delta,
             mu_d = float(10**(-10)),
             mu_u = float(10**10),
             max_iter = 100,
             zero_tol = 10**(-15)):
    P = P_cdom_persp(x, eta)
    l = -delta + f_persp(P[0],P[1])
    prox_mu = lambda mu : prox_f_persp(x,eta, mu)
    phi = lambda mu : mu + delta - f_persp_prox(x,eta,mu)
    if l <= 0:
        return [P[0],P[1],delta]
    else:
        if eta > 0 or (eta <= 0 and x <= 0) :
            mu_u = l
        else: 
            aux = phi(mu_u)
            while aux <= 0.0:
                if abs(aux) <= zero_tol:
                    aux_prox = prox_mu(mu_u)
                    return [aux_prox[0],aux_prox[1],delta + mu_u]
                mu_u *= 10
                aux = phi(mu_u)
            aux = phi(mu_d)
            print("aux = ", aux)
            while aux >= 0.0:
                if abs(aux) <= zero_tol:
                    aux_prox = prox_mu(mu_d)
                    return [aux_prox[0],aux_prox[1],delta + mu_d]
                mu_d *= 10**(-1)
                aux = phi(mu_d)
        if abs(phi(mu_u)) <= zero_tol:
            aux_prox = prox_mu(mu_u)
            return [aux_prox[0],aux_prox[1],delta + mu_u]
        if abs(phi(mu_d)) <= zero_tol:
            aux_prox = prox_mu(mu_d)
            return [aux_prox[0],aux_prox[1],delta + mu_d]
        print("phi(mu_d) = ",phi(mu_d))
        print("phi(mu_u) = ",phi(mu_u))
        mu_star = bisect(phi,mu_d,mu_u,maxiter = max_iter)
        aux_prox = prox_mu(mu_star)
        return [aux_prox[0],aux_prox[1],delta + mu_star]
#%%
x = [random.randint(-100, 100)*random.random() for i in range(0,1000)]
eta = [random.randint(-100, 100)*random.random() for i in range(0,1000)]
delta = [random.randint(-100, 100)*random.random() for i in range(0,1000)] 
t_0 = time.time()
for i in range(0,100):
    print ("\nx=",x[i],
            "\neta = ",eta[i],
            "\ndelta = ", delta[i],
            "\ni = ",i,
            )
    print("\nproj_epi(x,eta,delta) = ",proj_epi(x[i],
                                          eta[i],
                                          delta[i],
                                          max_iter = 700)
          )
t_f = time.time()
print("elapsed time =",t_f - t_0,"seconds")
#%%
print ("\nx=",x[26],
        "\neta = ",eta[26],
        "\ndelta = ", delta[26],
        "\ni = ",26,
      )
t_0 = time.time()
print("\nproj_epi(x,eta,delta) = ",proj_epi(x[26],
                                            eta[26],
                                            delta[26],
                                            max_iter = 500)
        )
t_f = time.time()
print("elapsed time =",t_f - t_0,"seconds")
#%%
x_0= -95.70690747688141 
eta_0 =  2.583175059523779 
delta_0 =  -87.42666679964313
print("\nproj_epi(x,eta,delta) = ",proj_epi(x_0,
                                            eta_0,
                                            delta_0,
                                            max_iter = 500)
      )
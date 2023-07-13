# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:43:43 2023

@author: Cristobal
"""
from f_persp_prox import f_persp_prox 
from prox_perspective import prox_f_persp
#import math as mt
import numpy as np
import proxop as pr
from prelim import P_cdom_star,P_cdom_persp,f_star,f_persp,prox_f_star,bounds
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar
import time
import random
#%%
N = 10000
n = 10**3
x_array = [random.randint(-n, n)*random.random() for i in range(0,N)]
eta_array = [random.randint(-n, n)*random.random() for i in range(0,N)]
Gamma_array = [random.randint(1, n)*random.random() for i in range(0,N)]
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
N = 10000
x_array = [random.randint(-10**3, 10**3)*random.random() for i in range(0,N)]
eta_array = [random.randint(-10**3, 10**3)*random.random() for i in range(0,N)]
Gamma_array = [random.randint(1, 10**4)*random.random() for i in range(0,N)]
times = []
t_0 = time.time()
f_list = []
for i in range(0,N):
    t_p = time.time()
    aux_f = f_persp_prox(x_array[i],eta_array[i],
                         Gamma_array[i],
                         0.1,
                         2,
                         alg = "brentq")
    f_list.append(aux_f)
    print ("\nx=",x_array[i],
            "\neta = ",eta_array[i],
            "\nGamma = ", Gamma_array[i],
            "\ni = ",i,
            "\nprox_f_persp(x,eta,Gamma) = ",aux_f
            )
    t_pf = time.time()
    times.append(t_pf-t_p)
t_f = time.time()
print("\nelapsed time =",t_f - t_0,"seconds")
print("\naverage elapsed time =","{:.2e}".format(sum(times)/len(times)),
      "seconds")
X = np.linspace(0, len(f_list),len(f_list))
plt.plot(X,f_list, color = "blue")
plt.show()

n = f_list.index(max(f_list))
print ("\nMax Value info: \n------------------------------- \n",
       "\nx=",x_array[n],
        "\neta = ",eta_array[n],
        "\nGamma = ", Gamma_array[n],
        "\ni = ",n,
        "\nf_persp_prox(x,eta,Gamma) = ",f_persp_prox(x_array[n],
                                                      eta_array[n],
                                                      1,
                                                      0.1,
                                                      2,
                                                      alg = "brentq")
        )
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:43:43 2023

@author: Cristobal
"""
#from prelim import P_cdom_star,P_cdom_persp,f_star,f_persp,prox_f_star,bounds
#from f_persp_prox import f_persp_prox 
#from prox_perspective import prox_f_persp
from epiproj import proj_epi
from friberg import proj_primalexpcone
#import math as mt
import numpy as np
#import proxop as pr
from matplotlib import pyplot as plt
#from scipy.optimize import root_scalar
import time
import random
#%%
N = 10000
n = 20
x = [random.randint(-n, n)*random.random() for i in range(0,N)]
eta = [random.randint(-n, n)*random.random() for i in range(0,N)]
delta = [random.randint(-n, n)*random.random() for i in range(0,N)] 
#%%
times = []
proj_us = []
t_0 = time.time()
for i in range(0,N):
    t_p = time.time()
# =============================================================================
#     print ("\nx=",x[i],
#             "\neta = ",eta[i],
#             "\ndelta = ", delta[i],
#             "\ni = ",i,
#             )
# =============================================================================
    P = proj_epi(x[i],eta[i],delta[i],1,2)
    t_pf = time.time()
    times.append(t_pf-t_p)
    proj_us.append(np.array(P))
    print("\nproj_epi(x,eta,delta) = ",P)
t_f = time.time()
print("\nelapsed time =",t_f - t_0,"seconds")
print("\naverage elapsed time =","{:.2e}".format(sum(times)/len(times)),
      "seconds")
#%%
v0 = [np.array([x[i],eta[i],delta[i]]) for i in range(0,N)]
times = []
proj_f = []
t_0 = time.time()
for i in range(0,N):
    tp = time.time()
    P_f = proj_primalexpcone(v0[i])
    tpf = time.time()
    times.append(tpf-tp)
    proj_f.append(P_f[0])
    print("\nproj_epi(x,eta,delta) = ",P_f)
t_f = time.time()
print("\nelapsed time =",t_f - t_0,"seconds")
print("\naverage elapsed time =","{:.2e}".format(sum(times)/len(times)),
      "seconds")
#%%
#ERRORS
diff = np.array(proj_us) - np.array(proj_f)
errors = np.linalg.norm(diff,ord=2,axis = 1)
#%%
# No axis mentioned, so works on entire array
max_index = np.argmax(errors)
print("\nMax element index: ", max_index)
print("\nMax element : ", errors[max_index])
errors[np.argmax(errors)]
err_2 = np.delete(errors, max_index)
#%%
max_index2 = np.argmax(err_2)
print("\nMax element index: ", max_index2)
print("\nMax element : ", err_2[max_index2])
#%%
plt.plot(np.delete(errors, max_index))
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 12:06:35 2023

@author: Cristobal
"""
import numpy as np

def K_exp(x,y):
    return y*(np.exp(x/y))

def K_exp_x(x,y):
    return np.exp(x/y)

def K_exp_y(x,y):
    return (np.exp(x/y)*(y-x))/y

def K_exp_grad(x,y):
    return np.array([K_exp_x(x,y), K_exp_y(x,y),-1])

def normal(x,y,z,t):
    return np.array([x,y,z]) +t*K_exp_grad(x,y) 

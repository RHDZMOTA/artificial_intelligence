# -*- coding: utf-8 -*-
'''
@author: Rodrigo Hernández Mota
'''

import numpy as np

# Función y = x1^2 + x2^2 
def func_2v(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    # evaluate and return 
    return X[0] ** 2 + X[1] ** 2
    
# Función  y = -x1*(10+100*x1) - x2*(5+40*x1) - x3*(5+50*x3)
def func1_3v(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    # evaluate and return
    return -X[0]*(10+100*X[0]) - X[1]*(5+40*X[1]) - X[2]*(5+50*X[2]);
    
# Función y = ...
def func2_3v(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    # evaluate and return 
    u = X[0]**2-2*X[0]+1-10*np.cos(X[0]-1)+X[1]**2+X[1]+\
    0.25-10*np.cos(X[0]+0.5)+X[2]**2-10*np.cos(X[2])
    return u

# Función de Rast...
def func_rast(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    # evaluate and return 
    for i in range(len(X)):
        u = 10+ X[i]**2-10*np.cos(5*X[i])
    return u

# Función de Ackley
def func_ackley(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    # evaluate and return 
    for i in range(len(X)):
        u = -10*np.exp(-np.sqrt(X[i]**2)) - np.exp(np.cos(5*X[i]))
    return u

# Función de Rosen
def func_rosen(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    # evaluate and return 
    for i in range(len(X)-1):
        u = 100 * (X[i+1]-X[i])**2+(1-X[i+1])**2
    return u

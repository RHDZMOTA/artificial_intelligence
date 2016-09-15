# -*- coding: utf-8 -*-
'''
@author: Rodrigo Hernández Mota
'''

import numpy as np

def f4(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x

    xm = np.matrix(X)
    xm = xm.T
    
    # costo 
    c = np.matrix([170, 160, 175, 180, 195])
    c = c.T
    
    # función de gasto
    z = xm.dot(c)
    
    # reestric.
    nn = xm.shape[0]
    r = np.zeros((nn,10))
    alpha = 10000
    
    if len(a) == 1:
        # 0
        index = (-X[0] + 48 > 0)
        r[:,0] = alpha * index * (-X[0] + 48)
        # 1
        index = (-X[1] - X[0] + 79 > 0)
        r[:,1] = alpha * index * (-X[1] - X[0] + 79)
        # 2
        index = (65 - X[0] - X[1] > 0)
        r[:,2] = alpha * index * (65 - X[0] - X[1])
        #3 
        index = (87 - X[0] - X[1] - X[2] > 0)
        r[:,3] = alpha * index * (87 - X[0] - X[1] - X[2])
        # 4
        index = (64 - X[1] - X[2] > 0)
        r[:,4] = alpha * index * (64 - X[1] - X[2])
        # 5
        index = (73 - X[2] - X[3] > 0)
        r[:,5] = alpha * index * (73 - X[2] - X[3])
        #6 
        index = (82 - X[2] - X[3] > 0)
        r[:,6] = alpha * index * (82 - X[2] - X[3])
        #7
        index = (43 - X[3] > 0)
        r[:,7] = alpha * index * (43 - X[3])
        #8
        index = (52 - X[3] - X[4] > 0)
        r[:,8] = alpha * index * (52 - X[3] - X[4])
        #9
        index = (15 - X[4] > 0)
        r[:,9] = alpha * index * (15 - X[4])
    
    re = np.sum(r, axis = 1)
    z = np.array(z.T) + re
    return z[0]

# Función ej3 (2 variables)
def f3(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    
    # comienza fun
    z = []
    
    s = np.size(X[0])
    rest1 = np.zeros(s)
    rest2 = np.zeros(s)
    for i,j in zip(X[0], X[1]):
        if i+j >= 1:
            z.append(i**2 + j**2)
        else:
            z.append(-10*(i+j)+20)
    
    a1 = 10000
    a2 = 10000
    
    if len(a) == 1:
        index = (-X[0]-1>= 0)
        rest1[index] = -X[0][index]-1
        index = (-X[1]-1>= 0)
        rest2[index] = -X[1][index]-1
    
    
        
    return z + a1 * rest1 + a2 * rest2
    
    
    
    

# Función y = x^1
def func_1v(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    # evaluate and return 
    return X[0] ** 2

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

# Markowitz function
def markowitz_1(X):
    '''
    function that applies the markowitz portfolio simulation
    '''

    # download data and calc. parameters
    import findata as fd
    acc = ["GRUMAB.MX","BIMBOA.MX","SORIANAB.MX"]
    price, returns = fd.download(acc)
    pa, rst, cst = fd.parameters(price, returns)
    
    # conditional for genetic algo... 
    aa = np.shape(X[0])
    if len(aa) > 1:
        aux = X[0]
        for i in range(1,len(X)):
            aux = np.hstack([aux,X[i]])
        X = np.transpose(aux)
        
    # use part as matrix
    X = np.transpose(np.matrix(X))
    
    # determines the returns...  
    rp = X * np.transpose(rst)
    
    # determines the risk (var)
    riskp = X*cst*np.transpose(X)
    n,t = np.shape(X)
    te = np.arange(0,n)
    riskp = np.transpose(riskp[te,te])
    

    return np.array(rp), np.array(riskp)

def markowitz_2(rp, riskp):
    '''
    This function takes the two results of markowitz and...
    '''
    z = -rp + riskp
    return z

def markowitz(x):
    # determine if x is data for gen.algo (==2) or gen.pso
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    
    rp, riskp = markowitz_1(X)
    z = markowitz_2(rp, riskp)
    
    return z
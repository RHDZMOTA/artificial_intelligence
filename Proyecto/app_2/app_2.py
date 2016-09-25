# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 21:36:24 2016

@author: Rodrigo


General procedure:

Genetic_algorithm
    optimize: function h(f)
    f is the vector or 'time intervals'
    h(f) = PSO
        optimize: function z(x)
        x is the vector or weights
            z(x) contains function prom_mov()
        
"""
import numpy as np
import matplotlib.pyplot as plt
import func_psof as pso
import gen_fun as fu
import func_agf as ga
import findata as fd

# %%Number assets for portfolio 

nvar = 3

# %% Descarga de datos 
acc = ["GFNORTEO.MX","LIVEPOLC-1.MX","HERDEZ.MX","BIMBOA.MX", "SANMEXB.MX", "ALSEA.MX"]
#n_acc, temp = np.shape(X)
acc = acc[:nvar]# acc[:n_acc]
price, returns = fd.download(acc)


# %% Función para evaluar PSO

def psoeval_general(nvar, desc, f, price, returns):
    print('EVALUATING PSO ALGORITHM... \n')    
    # número de partículas
    npart = 100
    # parámetros de movimiento
    c1 = 0.01; c2 = 0.01
    # evalucación y slgorítmos pso
    prtl_mg, fpg, fp, prtl = pso.algo_psof(npart, c1, c2, fu.prom_mov, nvar, desc, f, price, returns)
    print('RESULTS: \n\n')
    # mostrar resultados. 
    st = 'mínimo'
    if desc == 1: st = 'máximo'
    string = '\n'
    for i in range(nvar):
        string = string+'    x'+str(i)+' = '+str(prtl_mg[i])+'\n'
        
    
    print('El',st,'encontrado es de',fpg, 'en la región {x0,x1,...,xn} :',string)
    
    
    return prtl_mg

def h(x, price, returns):
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    a,b,c = np.shape(X)
    X = np.reshape(X,(a,b))
    mX = np.asmatrix(X)
    
    h = {}
    for i in range(mX.shape[1]):
        fv = [int(mX[i,0]) for i in range(mX.shape[0])]
        prtl_mg = psoeval_general(3,-1, fv,price, returns)
        rport, sigma_port = fu.port_val(fv,prtl_mg,price, returns)
        h[(rport, sigma_port)] = [fv, prtl_mg]
    
    
    return h, 

xmin = 1; xmax = 128; delta= xmin-xmax; nbits=7;
npob = 10; npadres = 4;
pobl, yp, yprom = ga.gen_algo(xmin, delta, nbits, npob, npadres, nvar, psoeval_general, 1, 3)    

#psoeval_general(3, -1, [10,20,30], price, returns)




    
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 09:22:29 2016

@author: Rodrigo Hernández Mota (if693056)

Instrucciones:
* Poner en el mismo directorio el archivo llamado 'func_ag.py'
* [ agregar ]
"""

import numpy as np
import matplotlib.pyplot as plt
import func_ag as ga

# %% Funciones disponibles...

# Función y = x1^2 + x2^2 
def func_2v(X):
    yp = X[0][0] ** 2 + X[1][0] ** 2;
    return yp
    
# Función  y = -x1*(10+100*x1) - x2*(5+40*x1) - x3*(5+50*x3)
def func1_3v(X):
    yp = -X[0][0]*(10+100*X[0][0]) - X[1][0]*(5+40*X[1][0]) - X[2][0]*(5+50*X[2][0]);
    return yp
    
# Función y = ...
def func2_3v(X):
    yp = X[0][0]**2-2*X[0][0]+1-10*np.cos(X[0][0]-1)+X[1][0]**2+X[1][0]+0.25-10*np.cos(X[0][0]+0.5)+X[2][0]**2-10*np.cos(X[2][0])
    return yp

# Función de Rast...
def func_rast(X):
    for i in range(len(X)):
        z = 10+ X[i][0]**2-10*np.cos(5*X[i][0])
    return z

# Función de Ackley
def func_ackley(X):
    for i in range(len(X)):
        z = -10*np.exp(-np.sqrt(X[i][0]**2)) - np.exp(np.cos(5*X[i][0]))
    return z

# Función de Rosen
def func_rosen(X):
    for i in range(len(X)-1):
        z = 100 * (X[i+1][0]-X[i][0])**2+(1-X[i+1][0])**2
    return z



# %% Función para evaluación general 

def eval_general(func, nvar, desc):
    '''
    Evaluación genérica de funciones...
    
    '''
    # datos generales
    # número de pobladores y método de cruzamiento
    npob = 600; met_cruz = 1;
    # bits para discretizar región de búsqueda
    nbits = 11;
    # región de búsqueda
    xmin = -150; delta = 300;
    # número de padres
    npadres = 20
    
    # use the genetic algo.
    pobl, yp, yprom = ga.gen_algo(xmin, delta, nbits, npob, npadres, nvar, func, desc, met_cruz)
    
    # results in a dict.
    results = {}
    c = 0
    aux = []
    for j in range(nvar):
        aux.append(pobl[j][0])
    for i in yp:
        aux2 = []
        for k in range(nvar):
            aux2.append(aux[k][c])
        results[np.double(i)] = aux2
        c += 1
    
    print('With an xmin of :',xmin, 'and a delta of: ', delta)
    
    # to print results
    if desc == 1:
        valu = max(results.keys())
        indx = list(results.keys()).index(valu)
        xval = list(results.values())[indx]
        print('The max. value found was: ', valu)
        print('Region: ',xval)
    else: 
        valu = min(results.keys())
        indx = list(results.keys()).index(valu)
        xval = list(results.values())[indx]
        print('The min. value found was: ', valu)
        print('Region: ',xval)
        
    # plot resutls.
    #plt.plot(yprom)
    #plt.show()
    return 1

# %% Probar métodos de cruzamiento
ind = 1
cromx = '11111111111'
cromy = '00000000000'
nbits = 11

# un punto cruce
a = ga.un_pc(ind,cromx,cromy,nbits)
print('Un punto cruce:',a)

# doble punto cruce
b = ga.do_pc(ind,cromx,cromy,nbits)
print('Doble punto cruve:',b)

# cruzamiento uniforme
c = ga.cr_un(ind,cromx,cromy,nbits)
print('Cruzamiento uniforme:',c)

# cruzamiento aritmético
d = ga.cr_ar(ind,cromx,cromy,nbits)
print('Cruzamiento aritmético:',d)

# %% Ejemplos 

# Función de clase
#print('\n\n---------------------------------------------------')
#print('Función 01 vista en clase de dos variables (func_2v)')
#eval_general(func_2v,2,-1)

# Función de tarea01 ejercicio 01
print('\n\n---------------------------------------------------')
print('Tarea ejercicio 01 tres variables: maximizar (func1_3v)')
eval_general(func1_3v,3,1)

# Función de tarea01 ejercicio 02
print('\n\n---------------------------------------------------')
print('Tarea ejercicio 02 tres variables: minimizar (func2_3v)')
eval_general(func2_3v,3,-1)


'''
R E S U L T A D O S

Ejercicio 01
Valor máximo: 0.140327129267
X = {-0.09770396, -0.09770396, -0.09770396}

Ejercicio 02
Valor mínimo: -22.4957320068
X = {-0.09770396, -0.48851979, -0.09770396}
'''
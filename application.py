# -*- coding: utf-8 -*-
"""
Datos del equipo:
* Mariana Aragón (if695956)
* Horacio Enriquez (if696451)
* Daniela Guerra  (if693394)
* Rodrigo Hernández Mota (if693056)


Instrucciones:
* Poner en el mismo directorio el archivo llamado 'func_ag.py'
* [ agregar ]
"""

import numpy as np
import matplotlib.pyplot as plt
import func_ag as ga
import func_pso as pso
import gen_fun as f

# %% Función para evaluación general 

def gaaeval_general(func, nvar, desc):
    '''
    Evaluación genérica de funciones para algorítmo genético
    '''
    print('GENETIC ALGO. OPTIM. \n')

    # GENERAL PARAMETERS
    # número de pobladores y método de cruzamiento
    npob = 60; met_cruz = 3;
    # bits para discretizar región de búsqueda
    nbits = 10;
    # región de búsqueda
    xmin = -1; delta = 2;
    # número de padres
    npadres = 20
    
    # USE GENETIC ALGORITHM FOR OPTIMIZATION
    pobl, yp, yprom = ga.gen_algo(xmin, delta, nbits, npob, npadres, nvar, func, desc, met_cruz)
    
    # BEFORE PRINTING THE RESULTS...
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
    
    # PRINT RESULTS
    if desc == 1:
        valu = max(results.keys())
        minmax = 'max.'
    else:
        valu = min(results.keys())
        minmax = 'min'

    indx = list(results.keys()).index(valu)
    xval = list(results.values())[indx]
    
    rs = '\n'
    for i in range(len(xval)):
        aux = '    x'+str(i)+' = '+str(xval[i][0])+'\n'
        rs = rs + aux
    
    print('The',minmax,'value found was:', valu,'in the region {x0,x1,...,xn} : \n',rs)
    
    # PLOT RESUTLS (Average)  
    plt.plot(yprom)
    plt.title('Average performance per generation')
    plt.xlabel('Number of generations')
    plt.ylabel('Average performance (function evaluated)')
    plt.show()
    
    if nvar == 1:      
        x = np.arange(xmin,xmin+delta,delta/1000)
        plt.plot(x,func([x]),'b-',pobl[0][0],yp,'rx')
        plt.title('Function and population')
        plt.xlabel('x values')
        plt.ylabel('y values')
        plt.show()
    if nvar == 2:
        plt.plot(pobl[0][0], pobl[1][0],'rx',0,0,'bo')
        plt.title('Población y orígen')
        plt.xlabel('X0')
        plt.ylabel('X1')
        plt.show()
    
    return 1

#%% PSO

def psoeval_general(func, nvar, desc):
    print('PSO METHOD \n')    
    # número de partículas
    npart = 1000
    # parámetros de movimiento
    c1 = 0.01; c2 = 0.01
    # evalucación y slgorítmos pso
    prtl_mg, fpg, fp, prtl = pso.algo_pso(npart, c1, c2, func, nvar, desc)
    
    # mostrar resultados. 
    st = 'mínimo'
    if desc == 1: st = 'máximo'
    string = '\n'
    for i in range(nvar):
        string = string+'    x'+str(i)+' = '+str(prtl_mg[i])+'\n'
        
    
    print('El',st,'econtrado es de',fpg, 'en la región {x0,x1,...,xn} :',string)
    
    if nvar == 1:
        x = np.arange(-100,100,0.5)
        plt.plot(x,func([x]),'b-',prtl_mg,fpg,'ro')
        plt.title('Function to optimize and particles')
        plt.xlabel('Particles')
        plt.ylabel('Value')
        plt.show()
    if nvar == 2:
        plt.plot(prtl[0],prtl[1],'rx',prtl_mg[0],prtl_mg[1],'ro',0,0,'b.')
        
    return 1
    
# %% Ejemplos 

# Función de cuadrática de dos variables
print('\n\n---------------------------------------------------')
print('Minimizar la función: func_1v de 1 variable\n')
#psoeval_general(f.func_1v,1,-1)
#gaaeval_general(f.func_1v,1,-1)


# Función de cuadrática de dos variables
print('\n\n---------------------------------------------------')
print('Minimizar la función: func_2v de 2 variables\n')
#psoeval_general(f.func_2v,2,-1)
#gaaeval_general(f.func_2v,2,-1)

# Función de cuadrática de dos variables
print('\n\n---------------------------------------------------')
print('Optimizar función de markowitz para 3 activos\n')
psoeval_general(f.markowitz,3,-1)
gaaeval_general(f.markowitz,3,-1)

'''
# Función de tarea01 ejercicio 01
print('\n\n---------------------------------------------------')
print('Tarea ejercicio 01 tres variables: maximizar (func1_3v)\n')
psoeval_general(f.func1_3v,3,1)
gaaeval_general(f.func1_3v,3,1)

# Función de tarea01 ejercicio 02
print('\n\n---------------------------------------------------')
print('Tarea ejercicio 02 tres variables: minimizar (func2_3v)\n')
psoeval_general(f.func2_3v,3,-1)
gaaeval_general(f.func2_3v,3,-1)

'''
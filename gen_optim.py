# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:48:31 2016

@author: Rodrigo
"""

# optim vect

import numpy as np;
import matplotlib.pyplot as plt;
import func_ag as ga

#%%
# Aplicación de algorítmos genéticos como método de optimizacion para más de
# una variable (dos variables)
    
# población y bits... 
npob = 20;
nbits = 10;

# rango de búsqueda
xmin = 0;
delta = 1000
xmax = xmin+delta;
# padres e hijos
npadres = 10;
nhijos  = npob - npadres;
# número de iteraciones y promedio.
niter = 100;
yprom = np.zeros(niter);
nvar = 2
desc=1 # max o min

# inicio de población en varialbe pobl
# pobl[i] contiene la variable i
# pobl[i][0] contiene los valores variable i 
# pobl[i][1] contiene los valores en int de variable i
pobl = []
for i in range(0,nvar):
    pobl.append(ga.initpob(npob,nbits,xmin,xmax))
x1p,x1i = ga.initpob(npob,nbits,xmin,xmax)
x2p,x2i = ga.initpob(npob,nbits,xmin,xmax)

# ciclo 
for ind in range(0,niter):
    # evaluación
    yp = pobl[0][0] ** 2 + pobl[1][0] ** 2;
    yprom[ind]=np.mean(yp)
        
    # selección de los mejores padres (minimizar)
    padr=[]
    for i in range(0,nvar):
        padr.append(ga.selecbest(npadres,pobl[i][0],pobl[i][1],yp,desc))
    #x1pad,x1padi = ga.selecbest(npadres,x1p,x1i,yp,1)
    #x2pad,x2padi = ga.selecbest(npadres,x2p,x2i,yp,1)
    
    # generación de hijos
    hijs=[]
    for i in range(0,nvar):
        hijs.append(ga.cruzamiento(padr[i][1],nhijos,nbits,xmin,xmax))
    #x1hij,x1hiji=ga.cruzamiento(x1padi,nhijos,nbits,xmin,xmax)
    #x2hij,x2hiji=ga.cruzamiento(x2padi,nhijos,nbits,xmin,xmax)
    
    # mutaciones en hijos
    if ind%10==0:
        for i in range(0,nvar):
            hijs[i] = ga.mutacion(hijs[i][1],1,nbits,xmin,xmax)
    
    #if ind%10==0:
    #    x1hij,x1hiji=ga.mutacion(x1hiji,1,nbits,xmin,xmax)
    #    x2hij,x2hiji=ga.mutacion(x2hiji,1,nbits,xmin,xmax)
        
    # sobreescribir los padres e hijos
    
    pobl = []
    for i in range(0,nvar):
        aux0 = np.concatenate((padr[i][0],hijs[i][0]))
        aux1 = np.concatenate((padr[i][1],hijs[i][1]))
        pobl.append([aux0,aux1])


plt.plot(yprom)
plt.show()

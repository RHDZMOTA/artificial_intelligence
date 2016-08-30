# -*- coding: utf-8 -*-
"""
Funciones proporcionadas por Rienamm para faciliar la aplicación de los
algorítmos genéticos:
- initpob()
- selectbest()
- cruzamiento()
- mutación()

Otras funciones

"""

import numpy as np;
import matplotlib.pyplot as plt;

#%%
# Definición de la función de inicialización
def initpob(npob,nbits,xmin,xmax):
    # xint = np.round(((2**nbits)-1)*np.random.rand(npob,1));
    #
    # npob = numero de pobladores
    # nbits = numero de bits del codigo genetico
    # xmin = limite inferior del espacio de busqueda
    # xmax = limite superior del espacio de busqueda
    xint = np.random.randint(2**nbits,size=(npob,1))
    xpob = ((xmax-xmin)/(np.double(2**nbits)-1))*xint+xmin;
    return xpob,xint

#%%
# Definición de la función de selección
def selecbest(npadres,xpob,xint,ypob,minmax):
    # Ordenar los pobladores según su desempeño.
    #
    # npadres = numero de padres que se quieren salvar
    # xpob = pobladores en el espacio de busqueda
    # xint = cromosoma en decimal
    # ypob = desempeño de los pobladores
    # minmax = 1 si se quiere maximizar y -1 si se quiere minimizar
    npob,ntemp = np.shape(xpob);
    temp = np.zeros((npob,3));
    temp[:,0] = ypob[:,0];
    temp[:,1] = xpob[:,0];
    temp[:,2] = xint[:,0];
    temp = temp[temp[:,0].argsort(),:];
    if minmax == 1:
        xpadres=temp[npob-npadres:npob,1];
        xintpadres=temp[npob-npadres:npob,2];
    elif minmax == -1:
        xpadres=temp[0:npadres,1];
        xintpadres=temp[0:npadres,2];
    else:
        print('No se eligio un argumento valido para minmax')
        return -1,1
    xpadres = xpadres.reshape((npadres,1));
    xintpadres = xintpadres.reshape((npadres,1));
    return xpadres,xintpadres

#%% Métodos de cruzamiento

# Un punto cruce 
def un_pc(ind,cromx,cromy,nbits):
    crombit = np.random.randint(nbits-1)+1;
    if ind%2 == 0:
        cromhijo = cromx[0:crombit]+cromy[crombit:nbits];
    else:
        cromhijo = cromy[crombit:nbits]+cromx[0:crombit];
    return cromhijo

# Doble punto cruce 
def do_pc(ind,cromx,cromy,nbits):
    i = np.random.randint(nbits//2-1)+1;
    j = nbits - (np.random.randint(nbits//2-2)+1);
    
    if ind%2 == 0:
        cromhijo = cromx[0:i]+cromy[i:j]+cromx[j:nbits]
    else:
        cromhijo = cromy[0:i]+cromx[i:j]+cromy[j:nbits]
    return cromhijo

# Cruzamiento uniforme
def cr_un(ind,cromx,cromy,nbits):
    cromhijo = ''
    for i in range(nbits):
        if np.random.rand(1) > 0.5:
            cromhijo = cromhijo + cromx[i]
        else:
            cromhijo = cromhijo + cromy[i]
    return cromhijo

# Cruzamiento aritmético 
def cr_ar(ind,cromx,cromy,nbits):
    x = int(cromx,2)
    y = int(cromy,2)
    z = x + y
    zbin = np.binary_repr(z,nbits)
    cromhijo = zbin[0:nbits]
    return cromhijo


#%%
#Definición de la función de cruzamiento
def cruzamiento(xint,nhijo,nbits,xmin,xmax,met_cruz):
    # Realizacion de nhijo numero de pobladores nuevos a partir de xint
    #
    # xint = cromosoma en decimal
    # nhijo = numero de hijos que se quieren generar
    # nbits = numero de bits del cromosoma de los pobladores
    # xmin = limite inferior del espacio de busqueda
    # xmax = limite superior del espacio de busqueda

    npadre,ntemp = np.shape(xint);
    xinthijo=np.zeros((nhijo,1));
    xhijo=xinthijo;
    if npadre==1:
        print("Precuación: Con un solo padre no se tiene recombinación de genes")
    px = np.tile(np.arange(0,npadre),nhijo//npadre+1);
    
    for ind in range(0,nhijo):
        cromx = np.binary_repr(np.int(xint[px[ind],0]),width=nbits);
        cromy = np.binary_repr(np.int(xint[px[ind+1],0]),width=nbits);
        
        if met_cruz == 1:
            # Metodo 1 para cruzamiento
            cromhijo = un_pc(ind,cromx,cromy,nbits)
        elif met_cruz == 2:
            #Metodo 2 para cruzamiento
            cromhijo = do_pc(ind,cromx,cromy,nbits)
        elif met_cruz == 3:
            #Metodo 3 para cruzamiento
            cromhijo = cr_un(ind,cromx,cromy,nbits)
        elif met_cruz == 4:
            #Metodo 4 para cruzamiento
            cromhijo = cr_un(ind,cromx,cromy,nbits)
            
        xinthijo[ind,0]=int(cromhijo,2);
        xhijo[ind,0] = ((xmax-xmin)/(np.double(2**nbits)-1))*xinthijo[ind,0]+xmin;
    return xhijo,xinthijo
    
#%%
#Definición de la función de cruzamiento
def mutacion(xint,nmut,nbits,xmin,xmax):
    # Realizacion de nmut numero de mutaciones en los pobladores xint
    #
    # xint = cromosoma en decimal
    # nmut = numero de mutantes que se queiren generar
    # nbits = numero de bits del cromosoma de los pobladores
    # xmin = limite inferior del espacio de busqueda
    # xmax = limite superior del espacio de busqueda
    nhijo,ntemp = np.shape(xint);
    for ind in range(0,nmut):
        nhijmut = np.random.randint(nhijo);
        nbitmut = np.random.randint(nbits);
        crom = np.binary_repr(np.int(xint[nhijmut,0]),width=nbits);
        if crom[nbitmut] == '1':
            crom = crom[0:nbitmut]+'0'+crom[nbitmut+1:nbits];
        else:
            crom = crom[0:nbitmut]+'1'+crom[nbitmut+1:nbits];
        xint[nhijmut,0]=int(crom,2);
    xmut = ((xmax-xmin)/(np.double(2**nbits)-1))*xint+xmin;
    return xmut,xint

# %%
    
def gen_algo(xmin, delta, nbits, npob, npadres, nvar, func, desc, met_cruz):
    '''
    La función gen_algo(...) aplica la metodología de optimización mediante
    algorítmos genéticos.
    
    I N P U T:
    - xmin: 
    - delta:
    - nbits:
    - npob: 
    - npadres:
    - nvar: 
    - func:
    - desc:
    - met_cruz:
    
    O U T P U T
    - 
    '''
    
    # Se determina el límite superior de la región  
    xmax = xmin+delta;
    
    # Se determina el número de hijos
    nhijos  = npob - npadres;
    
    # Número de iteraciones e inicialización del promedio.
    niter = 1000;
    yprom = np.zeros(niter);
    
    # Inicio de población en varialbe pobl
    # características: pobl es una lista que contiene listas.
    # ejemplo: pobl[i][j] accede a la lista j de la variable i+1 en donde
    # j solamente puede adquirir valores de {0, 1}
    # e.g. pobl[0][0] regresa el valor real de la variable 1.
    # e.g. pobl[0][1] regresa el valor entero de la variable 1.
    pobl = []
    for i in range(0,nvar):
        pobl.append(initpob(npob,nbits,xmin,xmax))
    
    # Ciclo para comenzar desarrollo y selección de la población 
    for ind in range(0,niter):
        # Evaluación del 'performance' de acuerdo al criterio func
        yp = func(pobl);
        yprom[ind]=np.mean(yp)
            
        # Selección de los individuos más aptos (futuros padres)
        padr=[]
        for i in range(0,nvar):
            padr.append(selecbest(npadres,pobl[i][0],pobl[i][1],yp,desc))
        
        # Recombinzación genética de los individuos con mejor desepeño.
        # (generación de hijos)
        hijs=[]
        for i in range(0,nvar):
            hijs.append(cruzamiento(padr[i][1],nhijos,nbits,xmin,xmax,met_cruz))
        
        # Mutación aleatoria cada 'n' generaciones 
        if ind%10==0:
            for i in range(0,nvar):
                hijs[i] = mutacion(hijs[i][1],4,nbits,xmin,xmax)
        
        # Eliminar a la población con bajo desempeño. 
        # Sobreescribir 'padres' e 'hijos'.
        pobl = []
        for i in range(0,nvar):
            aux0 = np.concatenate((padr[i][0],hijs[i][0]))
            aux1 = np.concatenate((padr[i][1],hijs[i][1]))
            pobl.append([aux0,aux1])
        
    return pobl, yp, yprom
    
#%% EJERCICIO 01
def ex1():
    # Aplicación con y**2
        
    # población y bits... 
    npob = 20;
    nbits = 10;
    
    # rangpo de búsqueda
    xmin = 0;
    xmax = 1023;
    # padres e hijos
    npadres = 10;
    nhijos  = npob - npadres;
    # número de iteraciones y promedio.
    niter = 100;
    yprom = np.zeros(niter);
    
    # inicio de población
    x1p,x1i = initpob(npob,nbits,xmin,xmax)
    
    for ind in range(0,niter):
        # evaluación
        yp = x1p ** 2;
        yprom[ind]=np.mean(yp)
        
        # selección de los mejores padres (minimizar)
        x1pad,x1padi = selecbest(npadres,x1p,x1i,yp,1)
        # generación de hijos
        x1hij,x1hiji=cruzamiento(x1padi,nhijos,nbits,xmin,xmax)
        # mutaciones en hijos
        if ind%10==0:
            x1hij,x1hiji=mutacion(x1hiji,1,nbits,xmin,xmax)
        
        # sobreescribir los padres e hijos
        x1p[0:npadres,0]=x1pad[:,0];
        x1p[npadres:npob,0]=x1hij[:,0];
        x1i[0:npadres,0]=x1padi[:,0];
        x1i[npadres:npob,0]=x1hiji[:,0];
        plt.plot(yprom)
        plt.show()
        return 1


#%% EJERCICIO 02
# Aplicación para más de una variable
def ex2():
    
    # población y bits... 
    npob = 20;
    nbits = 10;
    
    # rangpo de búsqueda
    xmin = 0;
    xmax = 1023;
    # padres e hijos
    npadres = 10;
    nhijos  = npob - npadres;
    # número de iteraciones y promedio.
    niter = 100;
    yprom = np.zeros(niter);
    
    # inicio de población
    x1p,x1i = initpob(npob,nbits,xmin,xmax)
    x2p,x2i = initpob(npob,nbits,xmin,xmax)
    
    for ind in range(0,niter):
        # evaluación
        yp = x1p ** 2 + x2p ** 2;
        yprom[ind]=np.mean(yp)
        
        # selección de los mejores padres (minimizar)
        x1pad,x1padi = selecbest(npadres,x1p,x1i,yp,1)
        x2pad,x2padi = selecbest(npadres,x2p,x2i,yp,1)
        # generación de hijos
        x1hij,x1hiji=cruzamiento(x1padi,nhijos,nbits,xmin,xmax)
        x2hij,x2hiji=cruzamiento(x2padi,nhijos,nbits,xmin,xmax)
        # mutaciones en hijos
        if ind%10==0:
            x1hij,x1hiji=mutacion(x1hiji,1,nbits,xmin,xmax)
            x2hij,x2hiji=mutacion(x2hiji,1,nbits,xmin,xmax)
        
        # sobreescribir los padres e hijos
        x1p[0:npadres,0]=x1pad[:,0];
        x1p[npadres:npob,0]=x1hij[:,0];
        x1i[0:npadres,0]=x1padi[:,0];
        x1i[npadres:npob,0]=x1hiji[:,0];
        x2p[0:npadres,0]=x2pad[:,0];
        x2p[npadres:npob,0]=x2hij[:,0];
        x2i[0:npadres,0]=x2padi[:,0];
        x2i[npadres:npob,0]=x2hiji[:,0];
    
    
    
    plt.plot(yprom)
    plt.show()
    return 1


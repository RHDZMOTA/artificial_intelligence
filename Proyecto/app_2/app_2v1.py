# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 21:36:24 2016

@author: Rodrigo


General procedure:

Genetic_algorithm
    optimize: function h(f,x)
    f is the vector or 'time intervals'
    x is the vector of weights 
    
        
"""

print('\nReading dependencies... \n\n')
import numpy as np
import matplotlib.pyplot as plt
import gen_fun as fu
import findata as fd

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
    temp[:,0] = ypob#ypob[:,0];
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
            cromhijo = cr_ar(ind,cromx,cromy,nbits)
            
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
    
def gen_algo(xmin, delta, nbits, npob, npadres, nvar, func, desc, met_cruz, price, returns):
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
    #xmax = xmin+delta;
    
    # Se determina el número de hijos
    nhijos  = npob - npadres;
    
    # Número de iteraciones e inicialización del promedio.
    niter = 50;
    yprom = np.zeros(niter);
    
    # Inicio de población en varialbe pobl
    # características: pobl es una lista que contiene listas.
    # ejemplo: pobl[i][j] accede a la lista j de la variable i+1 en donde
    # j solamente puede adquirir valores de {0, 1}
    # e.g. pobl[0][0] regresa el valor real de la variable 1.
    # e.g. pobl[0][1] regresa el valor entero de la variable 1.
    '''
    Para esta aplicación es necesario generar 6 fechas y 6 ponderaciones
    '''
    pobl1 = []
    pobl2 = []
    for i in range(0,int(nvar/2)):
        pobl1.append(initpob(npob,7,1,128))
        pobl2.append(initpob(npob,10,0,1))
    
    pobl = pobl1+pobl2
    
    # Ciclo para comenzar desarrollo y selección de la población 
    for ind in range(0,niter):
        # Evaluación del 'performance' de acuerdo al criterio func
        yp = func(pobl, price, returns);
        yprom[ind]=np.mean(yp)
            
        # Selección de los individuos más aptos (futuros padres)
        padr=[]
        for i in range(0,nvar):
            padr.append(selecbest(npadres,pobl[i][0],pobl[i][1],yp,desc))
        
        # Recombinzación genética de los individuos con mejor desepeño.
        # (generación de hijos)
        hijs1 = []
        hijs2 = []
        nhijos = npob - npadres#int((npob - npadres)/2)
        for i in range(0,int(nvar/2)):
            # 
            hijs1.append(cruzamiento(padr[i][1],nhijos,7,1,128,3))
            hijs2.append(cruzamiento(padr[i][1],nhijos,11,0,1,3))
        
        hijs = hijs1+hijs2
        
        # Mutación aleatoria cada 'n' generaciones 
        if ind%10==0:
            hijs1 = []
            hijs2 = []
            for i in range(0,int(nvar/2)):
                hijs1.append(mutacion(hijs[i][1],4,7,1,128))
                hijs2.append(mutacion(hijs[i][1],4,11,0,1))
            hijs = hijs1+hijs2
        
        # Eliminar a la población con bajo desempeño. 
        # Sobreescribir 'padres' e 'hijos'.
        pobl = []
        for i in range(0,nvar):
            aux0 = np.concatenate((padr[i][0],hijs[i][0]))
            aux1 = np.concatenate((padr[i][1],hijs[i][1]))
            pobl.append([aux0,aux1])
        
        print('Evaluating: {} %'.format(100*(ind+1) /niter))
    
    yp = func(pobl, price, returns);    
    return pobl, yp, yprom

# %%

def func(pobl, price, returns):
    x = pobl
    a = np.shape(x[0])
    if len(a) > 1: X = [x[i][0] for i in range(len(x))]
    else: X = x
    a,b,c = np.shape(X)
    X = np.reshape(X,(a,b))
    mX = np.asmatrix(X)
    
    
    #for i in range(int(np.shape(X)[0]/2)):
    #    X[i] = [int(j) for j in X[i]]
    br = int(np.shape(X)[0]/2)
    for i in range(mX.shape[1]):
        z = fu.prom_mov2(mX[br:, ].T,mX[:br, ].T,price, returns,1)
        
        
    return z


# %%Number assets for portfolio 

nacci = 6
nvar = nacci*2

# %% Descarga de datos 
acc = ["GFNORTEO.MX","LIVEPOLC-1.MX","HERDEZ.MX","BIMBOA.MX", "SANMEXB.MX", "ALSEA.MX"]
#n_acc, temp = np.shape(X)
acc = acc[:nvar]# acc[:n_acc]
price, returns = fd.download(acc)

xmin = 1; xmax = 128; delta= xmin-xmax; nbits=7;
npob = 40; npadres = 12;
desc = -1
print('Running genetic algorithm optimization... \n\n')
pobl, yp, yprom = gen_algo(xmin, delta, nbits, npob, npadres, nvar, func, desc, 3, price, returns)

print('\nShowing Results: ')



#%%
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

print('With an xmin of :',0, 'and a delta of: ', 1, 'and a window from:',1,' days to:',128,'days;')

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


# %%
'''
xval has te answer...

'''

print('\n\nResults in context: \n')

br = int(nvar/2)
mx = np.matrix(xval)
r, s = fu.prom_mov3g(mx[br:, ].T,mx[:br, ].T,price, returns,0)

print('\n\nReturn: {}'.format(r))
print('Standard Deviation: {}'.format(s))

print('\n sum: {}'.format(sum(mx[br:, ])))








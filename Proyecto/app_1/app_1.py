# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 21:36:24 2016

@author: Rodrigo
"""
import numpy as np
import matplotlib.pyplot as plt
import func_pso as pso
import gen_fun as f
import findata as fd

# %% función para evaluar 
def psoeval_general(func, nvar, desc):
    print('\n\nEvaluating pso-algorithm... \n')    
    # número de partículas
    npart = 2000
    # parámetros de movimiento
    c1 = 0.01; c2 = 0.01
    # evalucación y slgorítmos pso
    prtl_mg, fpg, fp, prtl = pso.algo_pso(npart, c1, c2, func, nvar, desc)
    print('RESULTS: \n\n')
    # mostrar resultados. 
    st = 'mínimo'
    if desc == 1: st = 'máximo'
    string = '\n'
    for i in range(nvar):
        string = string+'    x'+str(i)+' = '+str(prtl_mg[i])+'\n'
        
    
    print('El',st,'encontrado es de',fpg, 'en la región {x0,x1,...,xn} :',string)
    
    
    return prtl_mg
 
# %% gráficos de acciones 
print('\nReading data and dependencies...')
acc = ["GFNORTEO.MX","LIVEPOLC-1.MX","HERDEZ.MX","BIMBOA.MX", "SANMEXB.MX", "ALSEA.MX"]
price, returns = fd.download(acc)
fd.graf_prec(price,returns, acc)
   
# %% Encontrar proporciones de activos utilizando PSO
nact = 6
print('\n\n------------------------------------------------------------------')
print('Dado un portafolio de n = {} activos se determinan las proporciones '.format(nact))
print('de inversión de cada uno mediante optimización por PSO. \n\n')
x = psoeval_general(f.markowitz,nact,-1)


# %% Evaluar y graficar markowitz 
print('\n\n------------------------------------------------------------------')
print('Usando las ponderaciones determinadas por el algoritmo PSO, se calcula')
print('el rendimiento y riesgo del portafolio y se grafica junto con simulaciones')
print('de diferentes portafolios. \n\n')
X = np.array([[i] for i in x])
rp, riskp, X, pa     = f.markowitz_1(X)
rpo, riskpo, Xo, pao = f.markowitz_1(pa.T)

f1 = plt.figure()
plt.plot(riskpo, rpo, 'b.', riskp, rp, 'ro')
plt.title('Creación y selección de portafolios')
plt.xlabel('Riesgo del portafolio')
plt.ylabel('Rendimiento del portafolio')
#f1.axes.get_xaxis().set_ticks([])
#f1.axes.get_yaxis().set_ticks([])
plt.show(f1)

print('\n\nDatos del portafolio seleccionado:\n')
print('\t Rendimiento: {}'.format(rp[0][0]))
print('\t Riesgo: {}\n\n'.format(riskp[0][0]))






